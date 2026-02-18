#!/usr/bin/env python3
"""
SpriteGen Server
Pixel art generation using Stable Diffusion with LoRA support and BiRefNet background removal.
"""
import os
import sys
import json
import base64
import io
import gc
import warnings
from datetime import datetime

# Suppress warnings early — before torch/diffusers imports trigger autocast warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F

# Ensure xformers is importable (real or mock) so diffusers doesn't crash at import time.
# If xformers is missing or broken, inject a no-op mock into sys.modules.
# Diffusers' is_xformers_available() will still gate actual usage.
try:
    import xformers.ops
except (ImportError, RuntimeError, OSError, ValueError):
    import types, importlib
    _mock = types.ModuleType("xformers")
    _mock.__spec__ = importlib.machinery.ModuleSpec("xformers", None)
    _mock.ops = types.ModuleType("xformers.ops")
    _mock.ops.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", None)
    sys.modules["xformers"] = _mock
    sys.modules["xformers.ops"] = _mock.ops

from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np

from backends import detect_backend

# Suppress model-specific warnings
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModelWithProjection.*")
warnings.filterwarnings("ignore", message=".*torch.distributed.*")

# Suppress Flask development server warning
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

from flask_sock import Sock
sock = Sock(app)

AVAILABLE_MODELS = [
    "stabilityai/stable-diffusion-3.5-large-turbo",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5"
]

# Hub LoRAs tagged by model type
HUB_LORAS = [
    {"name": "nerijs/pixel-art-xl", "type": "xl"},
    {"name": "ntc-ai/SDXL-LoRA-slider.pixel-art", "type": "xl"},
    {"name": "nerijs/pixel-art-medium-128-v0.1", "type": "sd15"},
]


def _get_model_type(model_name):
    if not model_name:
        return None
    lower = model_name.lower()
    if "stable-diffusion-3" in lower:
        return "sd3"
    if "xl" in lower:
        return "xl"
    return "sd15"


def _get_available_loras(model_name):
    loras = ["None"]
    model_type = _get_model_type(model_name)
    for lora in HUB_LORAS:
        if lora["type"] == model_type:
            loras.append(lora["name"])
    if os.path.isdir("loras"):
        for f in os.listdir("loras"):
            if f.endswith(".safetensors"):
                loras.append(os.path.join("loras", f).replace("\\", "/"))
    return loras


class PixelArtSDServer:
    def __init__(self):
        self.pipeline = None
        self.segmentation_model = None
        self.segmentation_processor = None
        self.model_loaded = False
        self.current_model = None
        self.backend = detect_backend()
        self.device = self.backend.device
        self.model_cache = {}
        self.img2img_pipe = None
        self.offline_mode = False
        self.base_resolution = 512
        self.precision = torch.float16  # active dtype, updated per model

        self.default_settings = {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite"
        }

        print(f"SpriteGen Server v0.2.0")
        self.backend.print_startup_info()
        caps = self.backend.capabilities
        dtype_sample = caps.optimal_dtype("sd3")
        print(f"  Optimal dtype: {dtype_sample} (SD3.5)")

    def build_status_response(self):
        return {
            "server": "online",
            "model_loaded": self.model_loaded,
            "current_model": self.current_model,
            "model_type": _get_model_type(self.current_model),
            "device": self.device,
            "backend": self.backend.name,
            "gpu_name": self.backend.gpu_name,
            "vram_gb": self.backend.vram_gb,
            "available_models": AVAILABLE_MODELS,
            "available_loras": _get_available_loras(self.current_model),
            "version": "0.2.0"
        }

    def handle_generate_request(self, data):
        """Shared generation logic used by both REST and WebSocket endpoints."""
        prompt = data.get('prompt')
        if not prompt:
            raise ValueError("No prompt provided")

        print(f"\n  New generation request: {prompt[:30]}...")

        defaults = self.default_settings
        kwargs = {
            "lora_model": data.get('lora_model'),
            "lora_strength": data.get('lora_strength', 1.0),
            "num_inference_steps": data.get('steps', defaults.get('num_inference_steps')),
            "guidance_scale": data.get('guidance_scale', defaults.get('guidance_scale')),
            "seed": data.get('seed', -1),
            "negative_prompt": data.get('negative_prompt', defaults.get('negative_prompt')),
        }
        if 'width' in data:
            kwargs['width'] = data['width']
        if 'height' in data:
            kwargs['height'] = data['height']

        init_image = None
        if data.get('init_image'):
            img_data = data['init_image']
            raw_bytes = base64.b64decode(img_data['base64'])
            mode = 'RGBA' if img_data.get('mode') == 'rgba' else 'RGB'
            init_image = Image.frombytes(mode, (img_data['width'], img_data['height']), raw_bytes)
            kwargs['strength'] = data.get('strength', 0.7)
            print(f"  Input image: {img_data['width']}x{img_data['height']}, strength={kwargs['strength']}")

        start_time = datetime.now()
        image, used_seed = self.generate_image(
            prompt=prompt, init_image=init_image, **kwargs
        )
        gen_time = (datetime.now() - start_time).total_seconds()
        print(f"  Generation took {gen_time:.1f}s", flush=True)

        if data.get('remove_background', False):
            print("  Applying background removal...", flush=True)
            image = self.remove_background(image)
            print("  Background removal done", flush=True)

        pixel_width = int(data.get('pixel_width', 64))
        pixel_height = int(data.get('pixel_height', 64))
        colors = int(data.get('colors', 16))

        print(f"  Post-processing: {pixel_width}x{pixel_height}, {colors} colors...", flush=True)
        pixel_image = self.process_for_pixel_art(
            image, target_size=(pixel_width, pixel_height), colors=colors
        )
        del image
        print("  Pixel art processing done", flush=True)

        print("  Encoding to base64...", flush=True)
        img_base64 = self.image_to_base64(pixel_image)
        del pixel_image
        print(f"  Base64 encoding done ({len(img_base64)} chars)", flush=True)

        self._cleanup_memory()

        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"  Sending response (total {generation_time:.1f}s)")

        return {
            "success": True,
            "image": {
                "base64": img_base64,
                "width": pixel_width,
                "height": pixel_height,
                "mode": "rgba"
            },
            "seed": used_seed,
            "prompt": prompt,
            "generation_time": generation_time
        }

    def load_segmentation_model(self):
        """Loads the BiRefNet model for professional background removal."""
        if self.segmentation_model and self.segmentation_processor:
            return True

        print("Loading BiRefNet model for background removal...")

        # Check and install required packages
        missing_packages = []
        try:
            import einops
        except ImportError:
            missing_packages.append("einops")

        try:
            import kornia
        except ImportError:
            missing_packages.append("kornia")

        # Auto-install missing packages
        if missing_packages:
            print(f"  Installing missing packages: {', '.join(missing_packages)}")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print(f"  Successfully installed: {', '.join(missing_packages)}")

                # Re-import after installation
                import einops
                import kornia
            except Exception as e:
                print(f"  Failed to auto-install packages: {e}")
                print(f"  Please manually run: pip install {' '.join(missing_packages)}")
                return False

        try:
            model_name = 'zhengpeng7/BiRefNet'

            # Create processor for image preprocessing
            self.segmentation_processor = transforms.Compose([
                transforms.Resize((352, 352), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # BiRefNet uses PyTorch — use torch_device (CPU for ONNX, cuda for CUDA/ROCm)
            seg_device = self.backend.torch_device

            # Load the segmentation model
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode
            )
            self.segmentation_model.to(seg_device)
            self.segmentation_model.eval()
            self._seg_device = seg_device

            print(f"  BiRefNet loaded on {seg_device}")
            return True

        except Exception as e:
            print(f"  Error loading BiRefNet model: {e}")
            return False

    def remove_background(self, pil_image):
        """Uses BiRefNet to create a high-quality transparency mask."""
        if not self.load_segmentation_model():
            raise Exception("Background removal model could not be loaded.")

        print("  Removing background with BiRefNet...")
        seg_device = getattr(self, '_seg_device', self.backend.torch_device)
        try:
            with torch.no_grad():
                # Convert to RGB for processing
                rgb_image = pil_image.convert("RGB")

                # Preprocess image
                input_tensor = self.segmentation_processor(rgb_image).unsqueeze(0).to(seg_device)

                # Generate mask
                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]

                # Resize mask to original image size
                mask = F.interpolate(logits, size=pil_image.size[::-1], mode='nearest')
                mask = torch.sigmoid(mask).squeeze()

                # Create binary mask with sharp edges for pixel art
                binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)

            # Apply mask to create transparent background
            mask_image = Image.fromarray(binary_mask * 255, mode='L')
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)

            print("  Background removal complete")
            return rgba_image

        except Exception as e:
            print(f"  Error during background removal: {e}")
            return pil_image.convert("RGBA")

    def _cleanup_memory(self):
        """Free GPU and CPU memory after heavy operations."""
        self.backend.cleanup_memory()

    def image_to_base64(self, image):
        """Convert PIL image to base64 encoded bytes."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return base64.b64encode(image.tobytes()).decode()

    def load_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """Load and cache AI models for efficient generation."""
        try:
            local_only = self.offline_mode
            if local_only:
                print("  Offline mode enabled: loading from cache only")

            # Check cache first
            if model_name in self.model_cache:
                print(f"  Loading {model_name} from cache...")
                self.pipeline = self.model_cache[model_name]
                self.current_model = model_name
                self.model_loaded = True
                return True

            # Evict previous model from cache to free RAM
            if self.model_cache:
                print("  Evicting previous model from cache...")
                self.model_cache.clear()
                self.img2img_pipe = None
                self.pipeline = None
                self._cleanup_memory()

            print(f"  Loading base model: {model_name}")
            model_type = _get_model_type(model_name)

            self.pipeline, self.precision = self.backend.load_pipeline(
                model_name, model_type, local_only
            )

            # Cache the model and invalidate img2img pipe
            self.model_cache[model_name] = self.pipeline
            self.img2img_pipe = None
            self.current_model = model_name
            self.model_loaded = True

            print(f"  Model loaded successfully: {model_name}")
            return True

        except Exception as e:
            print(f"  Error loading model {model_name}: {str(e)}")
            return False

    def _get_img2img_pipeline(self):
        """Create img2img pipeline from current txt2img pipeline.

        For PyTorch backends, from_pipe() shares the same nn.Module objects.
        Those modules already have CPU-offloading hooks installed — we only
        apply pipeline-level flags (tiling, slicing) which are NOT inherited.
        For ONNX backends, a separate pipeline is loaded from disk.
        """
        if self.img2img_pipe is None:
            print("  Creating img2img pipeline from loaded model...")
            model_type = _get_model_type(self.current_model)
            self.img2img_pipe = self.backend.load_img2img_pipeline(
                self.pipeline, model_type, self.precision
            )
            print("  img2img pipeline ready")
        return self.img2img_pipe

    def generate_image(self, prompt, init_image=None, strength=0.7,
                       lora_model=None, lora_strength=1.0, **kwargs):
        """Generate images with optional img2img input."""
        if not self.model_loaded:
            raise Exception("No base model loaded. Please load a model first.")

        # Check LoRA support
        if lora_model and lora_model.lower() not in ['none', ''] and not self.backend.supports_lora:
            print("  LoRA not supported with Windows ML backend, ignoring")
            lora_model = None

        try:
            mode = "img2img" if init_image else "txt2img"
            print(f"  [{mode}] Prompt: '{prompt[:50]}...'")

            pipeline_kwargs = {}

            # Handle LoRA
            if lora_model and lora_model.lower() not in ['none', '']:
                print(f"  Loading LoRA: {lora_model} (strength: {lora_strength})")
                try:
                    if os.path.exists(lora_model):
                        lora_path, weight_name = os.path.split(lora_model)
                        self.pipeline.load_lora_weights(lora_path, weight_name=weight_name)
                    else:
                        self.pipeline.load_lora_weights(lora_model)
                    pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
                except (ValueError, RuntimeError):
                    print(f"  LoRA incompatible with current model, generating without it")
                    lora_model = None

            gen_params = self.default_settings.copy()
            gen_params.update(kwargs)

            model_type = _get_model_type(self.current_model)
            is_sd3 = model_type == "sd3"
            is_turbo = "turbo" in self.current_model.lower()

            # Default base resolution from launcher config
            gen_params.setdefault('width', self.base_resolution)
            gen_params.setdefault('height', self.base_resolution)

            if is_turbo:
                gen_params['num_inference_steps'] = 4
                gen_params['guidance_scale'] = 0.0

            if "pixel art" not in prompt.lower():
                prompt += gen_params["pixel_art_prompt_suffix"]

            # Seed — always use CPU generator when CPU offloading is active
            # to avoid device mismatch with accelerate hooks
            seed = gen_params.get("seed", -1)
            generator = torch.Generator(device=self.backend.generator_device())

            if seed is not None and int(seed) != -1:
                generator.manual_seed(int(seed))
                print(f"  Using seed: {seed}")
            else:
                import random
                random_seed = random.randint(0, 2**32 - 1)
                generator.manual_seed(random_seed)
                print(f"  Using random seed: {random_seed}")
                seed = random_seed

            w, h = int(gen_params["width"]), int(gen_params["height"])

            pipeline_kwargs.update({
                "prompt": prompt,
                "num_inference_steps": int(gen_params["num_inference_steps"]),
                "guidance_scale": float(gen_params["guidance_scale"]),
                "generator": generator
            })

            if is_sd3:
                pipeline_kwargs["max_sequence_length"] = 256
            else:
                pipeline_kwargs["negative_prompt"] = gen_params["negative_prompt"]

            # Choose pipeline
            if init_image:
                pipe = self._get_img2img_pipeline()
                # Upscale input sprite to generation resolution
                resample = getattr(Image, 'Resampling', Image).LANCZOS
                upscaled = init_image.convert("RGB").resize((w, h), resample)
                pipeline_kwargs["image"] = upscaled
                pipeline_kwargs["strength"] = float(strength)
                print(f"  img2img: {w}x{h}, strength={strength}, {gen_params['num_inference_steps']} steps")
            else:
                pipe = self.pipeline
                pipeline_kwargs["width"] = w
                pipeline_kwargs["height"] = h
                print(f"  txt2img: {w}x{h}, {gen_params['num_inference_steps']} steps")

            print("  Running diffusion pipeline...", flush=True)
            with torch.inference_mode():
                result = pipe(**pipeline_kwargs)
            image = result.images[0]
            print("  Diffusion + VAE decode complete", flush=True)

            # Free pipeline result tensors and reclaim VRAM immediately
            del result
            self._cleanup_memory()
            print("  Memory cleanup done", flush=True)

            return image, seed

        finally:
            if lora_model and lora_model.lower() not in ['none', ''] and hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()

    def process_for_pixel_art(self, image, target_size=(64, 64), colors=16):
        """Advanced pixel art post-processing with color quantization."""
        # Pillow compat: NEAREST and MEDIANCUT may live in different places
        resample = getattr(Image, 'Resampling', Image).NEAREST
        mediancut = getattr(Image, 'Quantize', Image).MEDIANCUT

        # Resize with nearest neighbor for sharp pixels
        image = image.resize(target_size, resample)

        # Apply color quantization if specified
        if colors > 0:
            if image.mode == 'RGBA':
                alpha = image.getchannel('A')
                rgb_image = image.convert('RGB').quantize(
                    colors=int(colors) - 1, method=mediancut
                )
                image = rgb_image.convert('RGBA')
                image.putalpha(alpha)
            else:
                image = image.quantize(
                    colors=int(colors), method=mediancut
                ).convert('RGB')

        return image

# Initialize the server instance
sd_server = PixelArtSDServer()

@app.route('/generate', methods=['POST'])
def generate():
    """Main generation endpoint with comprehensive error handling."""
    try:
        data = request.get_json()
        result = sd_server.handle_generate_request(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify(sd_server.build_status_response())

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(sd_server.build_status_response())

@app.route('/load_model', methods=['POST'])
def load_model_route():
    try:
        data = request.get_json()
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({"success": False, "error": "No model_name provided"}), 400

        print(f"  Loading model: {model_name}")

        if sd_server.load_model(model_name):
            resp = sd_server.build_status_response()
            resp["success"] = True
            return jsonify(resp)
        else:
            return jsonify({"success": False, "error": f"Failed to load {model_name}"}), 500

    except Exception as e:
        print(f"  Model loading error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({"models": AVAILABLE_MODELS})

@app.route('/loras', methods=['GET'])
def list_loras():
    return jsonify({"loras": _get_available_loras(sd_server.current_model)})

@sock.route('/ws')
def websocket_handler(ws):
    import simple_websocket
    while True:
        try:
            raw = ws.receive(timeout=30)
        except simple_websocket.ConnectionClosed:
            break
        if raw is None:
            continue

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
            continue

        msg_type = msg.get("type")

        if msg_type == "status":
            resp = sd_server.build_status_response()
            resp["type"] = "status"
            ws.send(json.dumps(resp))

        elif msg_type == "generate":
            ws.send(json.dumps({"type": "generating"}))
            try:
                result = sd_server.handle_generate_request(msg)
                result["type"] = "result"
                ws.send(json.dumps(result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                ws.send(json.dumps({"type": "error", "error": str(e)}))

        else:
            ws.send(json.dumps({"type": "error", "error": f"Unknown message type: {msg_type}"}))


def main(default_model_to_load=None, offline=False, base_resolution=512):
    """Main server startup function."""
    print("\n" + "="*50)
    print("SPRITEGEN SERVER v0.2.0")
    print("="*50)

    sd_server.offline_mode = offline
    sd_server.base_resolution = int(base_resolution)

    if default_model_to_load and default_model_to_load.lower() != "none":
        print(f"  Loading default model: {default_model_to_load}")
        sd_server.load_model(default_model_to_load)
    else:
        print("  No model loaded on startup (will load on first request)")

    print("\n  Server Configuration:")
    print(f"   Host: 127.0.0.1")
    print(f"   Port: 5000")
    print(f"   Backend: {sd_server.backend.name}")
    print(f"   Base Resolution: {sd_server.base_resolution}x{sd_server.base_resolution}")
    print(f"   Offline Mode: {offline}")

    print("\n  Server ready! Access at http://127.0.0.1:5000")
    print("  SpriteGen plugin can now connect!")
    print("\n" + "="*50 + "\n")

    try:
        # Disable Flask development server warning
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None

        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n  Server shutting down gracefully...")
    except Exception as e:
        print(f"  Server error: {e}")

if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")
