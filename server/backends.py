"""
Backend abstraction for SpriteGen Server.

Provides a unified interface for CUDA, ROCm, Windows ML (ONNX+DirectML), and CPU
backends with hardware capability detection and optimal dtype selection.
"""
import gc
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
try:
    from diffusers import StableDiffusion3Img2ImgPipeline
except ImportError:
    StableDiffusion3Img2ImgPipeline = None


# ---------------------------------------------------------------------------
# Hardware capabilities
# ---------------------------------------------------------------------------

@dataclass
class HardwareCapabilities:
    """Detected hardware features used to select optimal inference settings."""
    # GPU identity
    arch_name: str = "unknown"       # "gfx1201", "sm_89", "cpu", etc.
    arch_family: str = "unknown"     # "rdna4", "rdna3", "ada", "ampere", "zen4", ...

    # Supported dtypes (ordered best -> worst)
    supports_fp8: bool = False       # E4M3FN native compute (RDNA4, Ada Lovelace)
    supports_bf16: bool = False      # Native bfloat16 (Ampere+, RDNA3+, Zen4+)
    supports_fp16: bool = True       # Float16 (virtually everything)
    supports_tf32: bool = False      # TensorFloat-32 (Ampere+, NVIDIA only)

    # Matrix acceleration
    has_tensor_cores: bool = False   # NVIDIA Tensor Cores (Volta+)
    has_wmma: bool = False           # AMD Wave Matrix Multiply (RDNA3+)
    has_xmx: bool = False            # Intel Xe Matrix Extensions (Arc)

    # CPU ISA features
    has_avx512: bool = False
    has_avx512_vnni: bool = False    # INT8 acceleration
    has_avx512_bf16: bool = False    # BF16 acceleration (Zen4, Alder Lake+)
    has_amx: bool = False            # Intel AMX (Sapphire Rapids+)

    # Memory
    vram_gb: float = 0.0

    def optimal_dtype(self, model_type: str = "sd3") -> torch.dtype:
        """Return best dtype for this hardware + model combination."""
        if self.supports_fp8 and model_type in ("sd3", "xl"):
            return torch.float8_e4m3fn
        if self.supports_bf16:
            return torch.bfloat16
        if self.supports_fp16:
            return torch.float16
        return torch.float32

    def summary_line(self) -> str:
        """One-liner for startup log."""
        parts = []
        for label, flag in [
            ("FP8", self.supports_fp8), ("BF16", self.supports_bf16),
            ("FP16", self.supports_fp16), ("TF32", self.supports_tf32),
            ("TensorCores", self.has_tensor_cores), ("WMMA", self.has_wmma),
            ("XMX", self.has_xmx), ("AVX-512", self.has_avx512),
            ("AMX", self.has_amx),
        ]:
            if flag:
                parts.append(label)
        return " | ".join(parts) if parts else "generic"


# ---------------------------------------------------------------------------
# Hardware detection functions
# ---------------------------------------------------------------------------

def _detect_nvidia_capabilities() -> HardwareCapabilities:
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)

    caps = HardwareCapabilities(
        arch_name=f"sm_{props.major}{props.minor}",
        vram_gb=props.total_memory / 1024**3,
    )

    if cc >= (8, 9):                  # Ada Lovelace (RTX 40xx)
        caps.arch_family = "ada"
        caps.supports_fp8 = True
        caps.supports_bf16 = True
        caps.supports_tf32 = True
        caps.has_tensor_cores = True
    elif cc >= (8, 0):                # Ampere (RTX 30xx, A100)
        caps.arch_family = "ampere"
        caps.supports_bf16 = True
        caps.supports_tf32 = True
        caps.has_tensor_cores = True
    elif cc >= (7, 5):                # Turing (RTX 20xx)
        caps.arch_family = "turing"
        caps.has_tensor_cores = True
    elif cc >= (7, 0):                # Volta
        caps.arch_family = "volta"
        caps.has_tensor_cores = True
    else:                             # Pascal or older
        caps.arch_family = "pascal"

    return caps


def _detect_amd_capabilities() -> HardwareCapabilities:
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "unknown")

    caps = HardwareCapabilities(arch_name=arch, vram_gb=props.total_memory / 1024**3)

    if arch.startswith("gfx12"):      # RDNA 4
        caps.arch_family = "rdna4"
        caps.supports_fp8 = True
        caps.supports_bf16 = True
        caps.has_wmma = True
    elif arch.startswith("gfx11"):    # RDNA 3
        caps.arch_family = "rdna3"
        caps.supports_bf16 = True
        caps.has_wmma = True
    elif arch.startswith("gfx103"):   # RDNA 2
        caps.arch_family = "rdna2"

    return caps


def _detect_cpu_capabilities() -> HardwareCapabilities:
    caps = HardwareCapabilities(arch_name="cpu")
    try:
        from cpufeature import CPUFeature
        caps.has_avx512 = CPUFeature.get("AVX512f", False)
        caps.has_avx512_vnni = CPUFeature.get("AVX512_VNNI", False)
        caps.has_avx512_bf16 = CPUFeature.get("AVX512_BF16", False)
        caps.has_amx = CPUFeature.get("AMX_TILE", False)

        if caps.has_avx512_bf16:
            caps.supports_bf16 = True
            caps.arch_family = "zen4"
        elif caps.has_avx512:
            caps.arch_family = "avx512"
        else:
            caps.arch_family = "generic"
    except ImportError:
        caps.arch_family = "generic"
    return caps


def _detect_windowsml_capabilities() -> HardwareCapabilities:
    caps = HardwareCapabilities(arch_name="windowsml", arch_family="windowsml")
    caps.supports_fp16 = True
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name"],
            capture_output=True, text=True, timeout=5,
        )
        names = [l.strip() for l in result.stdout.splitlines()
                 if l.strip() and l.strip() != "Name"]
        if names:
            name_lower = names[0].lower()
            if any(k in name_lower for k in ("9700", "9800", "9070")):
                caps.arch_family = "rdna4"
            elif any(k in name_lower for k in ("7900", "7800", "7700", "7600")):
                caps.arch_family = "rdna3"
    except Exception:
        pass
    return caps


def _gpu_name_from_wmi() -> str | None:
    """Get primary GPU name on Windows via WMI."""
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name"],
            capture_output=True, text=True, timeout=5,
        )
        names = [l.strip() for l in result.stdout.splitlines()
                 if l.strip() and l.strip() != "Name"]
        return names[0] if names else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Base backend class
# ---------------------------------------------------------------------------

class GPUBackend(ABC):
    """Abstract interface that every backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name, e.g. 'CUDA 12.1', 'Windows ML'."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Device identifier for status/display, e.g. 'cuda', 'windowsml', 'cpu'."""

    @property
    def torch_device(self) -> str:
        """Device string for PyTorch auxiliary ops (BiRefNet, tensors).
        Defaults to self.device; WindowsML overrides to 'cpu'."""
        return self.device

    @property
    @abstractmethod
    def gpu_name(self) -> str | None:
        """GPU model name or None."""

    @property
    @abstractmethod
    def vram_gb(self) -> float | None:
        """VRAM in GB or None."""

    @property
    @abstractmethod
    def supports_lora(self) -> bool:
        """Whether the backend supports LoRA loading."""

    @abstractmethod
    def select_dtype(self, model_type: str) -> torch.dtype:
        """Return the optimal dtype for the given model type."""

    @abstractmethod
    def setup_pipeline(self, pipe):
        """Configure a freshly-loaded txt2img pipeline (offloading, device placement).
        Returns the pipeline."""

    @abstractmethod
    def optimize_pipeline(self, pipe):
        """Apply pipeline-level optimizations (tiling, slicing) to a derived pipeline."""

    @abstractmethod
    def cleanup_memory(self):
        """Free backend-specific memory."""

    @abstractmethod
    def load_pipeline(self, model_name: str, model_type: str, local_only: bool = False):
        """Load a txt2img pipeline.  Returns (pipeline, precision)."""

    @abstractmethod
    def load_img2img_pipeline(self, parent_pipe, model_type: str, precision):
        """Create/load an img2img pipeline from the parent txt2img pipeline."""

    # Concrete helpers -------------------------------------------------------

    @property
    def capabilities(self) -> HardwareCapabilities:
        return self._capabilities

    def generator_device(self) -> str:
        """Device for torch.Generator — always CPU to avoid device-mismatch
        with accelerate CPU-offloading hooks."""
        return "cpu"

    def print_startup_info(self):
        """Print capability summary at server start."""
        caps = self._capabilities
        print(f"  Backend: {self.name}")
        if self.gpu_name:
            print(f"  GPU: {self.gpu_name}")
        if self.vram_gb is not None:
            print(f"  VRAM: {self.vram_gb:.1f}GB")
        if caps.arch_family != "unknown":
            print(f"  Architecture: {caps.arch_family} ({caps.arch_name})")
        print(f"  Capabilities: {caps.summary_line()}")

    @staticmethod
    def _apply_pipeline_flags(pipe):
        """Enable tiling + slicing flags (safe to call on any diffusers pipeline)."""
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()


# ---------------------------------------------------------------------------
# PyTorch-based backend mixin (shared by CUDA, ROCm, CPU)
# ---------------------------------------------------------------------------

class _PyTorchBackendMixin:
    """Shared load_pipeline / load_img2img_pipeline for diffusers-based backends."""

    def load_pipeline(self, model_name: str, model_type: str, local_only: bool = False):
        precision = self.select_dtype(model_type)
        print(f"  Using dtype: {precision}")

        if model_type == "sd3":
            print("  Loading SD 3.5 pipeline (without T5-XXL to save ~10GB)...")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_name,
                text_encoder_3=None, tokenizer_3=None,
                torch_dtype=precision, use_safetensors=True,
                local_files_only=local_only,
            )
        elif model_type == "xl":
            print("  Loading SDXL pipeline with optimized VAE...")
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=precision,
                local_files_only=local_only,
            )
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name, vae=vae,
                torch_dtype=precision, use_safetensors=True,
                local_files_only=local_only,
            )
        else:
            print("  Loading SD 1.5 pipeline...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=precision, use_safetensors=True,
                local_files_only=local_only,
            )

        pipe = self.setup_pipeline(pipe)
        self._apply_pipeline_flags(pipe)
        return pipe, precision

    def load_img2img_pipeline(self, parent_pipe, model_type: str, precision):
        """from_pipe() shares weights — no duplicate CPU-offloading hooks."""
        if model_type == "sd3" and StableDiffusion3Img2ImgPipeline is not None:
            pipe = StableDiffusion3Img2ImgPipeline.from_pipe(parent_pipe, torch_dtype=precision)
        elif model_type == "xl":
            pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(parent_pipe, torch_dtype=precision)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pipe(parent_pipe, torch_dtype=precision)
        self.optimize_pipeline(pipe)
        return pipe


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------

class CUDABackend(_PyTorchBackendMixin, GPUBackend):
    def __init__(self):
        self._capabilities = _detect_nvidia_capabilities()
        # Enable TF32 matmul if supported (free 2-3x speedup for single-precision)
        if self._capabilities.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @property
    def name(self) -> str:
        return f"CUDA {torch.version.cuda}"

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def gpu_name(self) -> str | None:
        return torch.cuda.get_device_name(0)

    @property
    def vram_gb(self) -> float | None:
        return self._capabilities.vram_gb

    @property
    def supports_lora(self) -> bool:
        return True

    def select_dtype(self, model_type: str) -> torch.dtype:
        return self._capabilities.optimal_dtype(model_type)

    def setup_pipeline(self, pipe):
        pipe.enable_model_cpu_offload()
        print("  Model CPU offloading enabled (auto VRAM management)")
        if self._capabilities.supports_tf32:
            print(f"  TF32 matmul enabled ({self._capabilities.arch_family})")
        return pipe

    def optimize_pipeline(self, pipe):
        self._apply_pipeline_flags(pipe)

    def cleanup_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ROCmBackend(_PyTorchBackendMixin, GPUBackend):
    def __init__(self):
        self._capabilities = _detect_amd_capabilities()

    @property
    def name(self) -> str:
        hip = getattr(torch.version, "hip", None)
        return f"ROCm (HIP {hip})" if hip else "ROCm"

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def gpu_name(self) -> str | None:
        return torch.cuda.get_device_name(0)

    @property
    def vram_gb(self) -> float | None:
        return self._capabilities.vram_gb

    @property
    def supports_lora(self) -> bool:
        return True

    def select_dtype(self, model_type: str) -> torch.dtype:
        return self._capabilities.optimal_dtype(model_type)

    def setup_pipeline(self, pipe):
        pipe.enable_model_cpu_offload()
        print("  Model CPU offloading enabled (auto VRAM management)")
        caps = self._capabilities
        if caps.supports_fp8:
            print(f"  RDNA 4 detected - FP8 WMMA available (ROCm 7.1+ required)")
        elif caps.has_wmma:
            print(f"  RDNA 3 detected - BF16/FP16 WMMA available")
        return pipe

    def optimize_pipeline(self, pipe):
        self._apply_pipeline_flags(pipe)

    def cleanup_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class WindowsMLBackend(GPUBackend):
    """ONNX Runtime + DirectML backend.
    Diffusion runs on GPU via DmlExecutionProvider.
    PyTorch (CPU-only) handles auxiliary ops like BiRefNet."""

    def __init__(self):
        self._capabilities = _detect_windowsml_capabilities()
        self._gpu_name = _gpu_name_from_wmi()

    @property
    def name(self) -> str:
        return "Windows ML"

    @property
    def device(self) -> str:
        return "windowsml"

    @property
    def torch_device(self) -> str:
        # PyTorch ops (BiRefNet, seeds) run on CPU;
        # diffusion inference runs on GPU via ONNX Runtime DmlExecutionProvider
        return "cpu"

    @property
    def gpu_name(self) -> str | None:
        return self._gpu_name

    @property
    def vram_gb(self) -> float | None:
        return None

    @property
    def supports_lora(self) -> bool:
        return False

    def select_dtype(self, model_type: str) -> torch.dtype:
        return torch.float16

    def setup_pipeline(self, pipe):
        return pipe

    def optimize_pipeline(self, pipe):
        pass

    def cleanup_memory(self):
        gc.collect()

    def load_pipeline(self, model_name: str, model_type: str, local_only: bool = False):
        escaped = model_name.replace("/", "--")
        onnx_dir = os.path.join("onnx_models", escaped)

        if not os.path.isfile(os.path.join(onnx_dir, "model_index.json")):
            raise RuntimeError(
                f"ONNX model not found at '{onnx_dir}'.\n"
                "Click 'Install Dependencies' in the launcher to convert."
            )

        from optimum.onnxruntime import ORTDiffusionPipeline

        labels = {"sd3": "SD 3.5", "xl": "SDXL", "sd15": "SD 1.5"}
        label = labels.get(model_type, model_type)

        print(f"  Loading {label} pipeline from '{onnx_dir}' (GPU: DmlExecutionProvider)...")
        pipe = ORTDiffusionPipeline.from_pretrained(
            onnx_dir, provider="DmlExecutionProvider",
        )
        print(f"  ONNX model loaded — inference will run on GPU via DirectML")
        return pipe, torch.float16

    def load_img2img_pipeline(self, parent_pipe, model_type: str, precision):
        model_dir = getattr(parent_pipe, "model_save_dir", None)
        if model_dir is None:
            raise RuntimeError(
                "Cannot create img2img pipeline: parent pipeline has no model_save_dir. "
                "Ensure the txt2img model was loaded from a local/cached directory."
            )

        # Try type-specific img2img classes first, fall back to reusing txt2img pipe
        img2img_classes = {
            "xl": "ORTStableDiffusionXLImg2ImgPipeline",
            "sd15": "ORTStableDiffusionImg2ImgPipeline",
            "sd3": "ORTStableDiffusion3Img2ImgPipeline",
        }

        cls_name = img2img_classes.get(model_type)
        if cls_name:
            try:
                import importlib
                mod = importlib.import_module("optimum.onnxruntime")
                PipeClass = getattr(mod, cls_name)
                print(f"  Loading img2img ONNX pipeline from cache (GPU: DmlExecutionProvider)...")
                return PipeClass.from_pretrained(
                    str(model_dir), provider="DmlExecutionProvider"
                )
            except (ImportError, AttributeError):
                pass
            except Exception as e:
                print(f"  img2img class {cls_name} failed: {e}")

        # Fallback: reuse the txt2img pipeline (caller passes strength for denoising)
        print(f"  img2img: reusing txt2img ONNX pipeline")
        return parent_pipe


class CPUBackend(_PyTorchBackendMixin, GPUBackend):
    def __init__(self):
        self._capabilities = _detect_cpu_capabilities()

    @property
    def name(self) -> str:
        return "CPU"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def gpu_name(self) -> str | None:
        return None

    @property
    def vram_gb(self) -> float | None:
        return None

    @property
    def supports_lora(self) -> bool:
        return True

    def select_dtype(self, model_type: str) -> torch.dtype:
        if self._capabilities.supports_bf16:
            return torch.bfloat16
        return torch.float32

    def setup_pipeline(self, pipe):
        pipe = pipe.to("cpu")
        caps = self._capabilities
        if caps.has_amx:
            print("  Intel AMX detected - accelerated INT8/BF16 inference")
        elif caps.has_avx512_bf16:
            print("  AVX-512 BF16 detected - accelerated BF16 inference")
        elif caps.has_avx512:
            print("  AVX-512 detected")
        return pipe

    def optimize_pipeline(self, pipe):
        self._apply_pipeline_flags(pipe)

    def cleanup_memory(self):
        gc.collect()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def detect_backend() -> GPUBackend:
    """Auto-detect the best available backend."""
    # 1. Check for ONNX Runtime with DirectML provider
    try:
        import onnxruntime as ort
        if "DmlExecutionProvider" in ort.get_available_providers():
            return WindowsMLBackend()
    except ImportError:
        pass

    # 2. CUDA with ROCm (HIP)
    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        if hip:
            return ROCmBackend()
        return CUDABackend()

    # 3. CPU fallback
    return CPUBackend()
