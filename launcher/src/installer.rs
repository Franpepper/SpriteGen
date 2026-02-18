use std::path::Path;

pub struct InstallStep {
    pub label: String,
    pub args: Vec<String>,
    pub allow_failure: bool,
}

fn step(label: &str, args: &[&str]) -> InstallStep {
    InstallStep {
        label: label.into(),
        args: args.iter().map(|s| s.to_string()).collect(),
        allow_failure: false,
    }
}

fn step_optional(label: &str, args: &[&str]) -> InstallStep {
    InstallStep {
        label: label.into(),
        args: args.iter().map(|s| s.to_string()).collect(),
        allow_failure: true,
    }
}

pub fn build_install_steps(venv_python: &Path, backend: &str, fresh_venv: bool) -> Vec<InstallStep> {
    let py = venv_python.to_str().unwrap_or("python");
    let mut steps = vec![];

    // Only uninstall old torch if there's an existing installation to clean up
    if !fresh_venv {
        steps.push(step_optional(
            "🗑️ Removing old PyTorch...",
            &[py, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "torch-directml", "onnxruntime-directml", "-y"],
        ));
    }

    // Step 2: Install PyTorch for selected backend
    match backend {
        "cuda" => {
            steps.push(step(
                "📦 Installing PyTorch (CUDA)...",
                &[
                    py, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu129",
                ],
            ));
        }
        "rocm" => {
            steps.push(step(
                "📦 Installing ROCm SDK...",
                &[
                    py, "-m", "pip", "install", "--no-cache-dir",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz",
                ],
            ));
            steps.push(step(
                "📦 Installing PyTorch (ROCm)...",
                &[
                    py, "-m", "pip", "install", "--no-cache-dir",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
                ],
            ));
        }
        "windowsml" => {
            steps.push(step(
                "📦 Installing PyTorch (CPU base)...",
                &[
                    py, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                ],
            ));
            // optimum-onnx provides optimum.exporters.onnx (conversion) and
            // optimum.onnxruntime (loading). onnxruntime-directml provides
            // the DmlExecutionProvider for GPU inference.
            steps.push(step(
                "📦 Installing Optimum ONNX...",
                &[py, "-m", "pip", "install", "optimum-onnx"],
            ));
            steps.push(step(
                "📦 Installing ONNX Runtime (DirectML)...",
                &[py, "-m", "pip", "install", "onnxruntime-directml"],
            ));
        }
        _ => {
            // CPU fallback
            steps.push(step(
                "📦 Installing PyTorch (CPU)...",
                &[
                    py, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                ],
            ));
        }
    }

    // Step 3: Other dependencies
    let mut deps: Vec<&str> = vec![
        py, "-m", "pip", "install",
        "diffusers>=0.27.0", "transformers>=4.55.5", "accelerate>=0.29.0",
        "flask>=2.3.0", "flask-cors>=4.0.0", "flask-sock>=0.7.0", "pillow>=9.5.0",
        "numpy>=1.24.0", "safetensors>=0.4.0", "peft>=0.11.0",
        "opencv-python", "scipy", "timm", "einops", "kornia",
        "cpufeature", "sentencepiece",
    ];
    if backend == "cuda" {
        deps.push("xformers>=0.0.20");
        deps.push("torchao>=0.4");
    }
    steps.push(step("📦 Installing AI libraries...", &deps));

    steps
}

/// Build the step to convert a model to ONNX format.
/// Called separately from the install flow (via the "Convert to ONNX" button).
pub fn build_onnx_convert_step(venv_python: &Path, model: &str, project_dir: &Path, dtype: &str) -> InstallStep {
    let py = venv_python.to_str().unwrap_or("python");
    let escaped = model.replace('/', "--");
    let onnx_dir = project_dir.join("onnx_models").join(&escaped);
    let script = project_dir.join("server").join("convert_onnx.py");
    let onnx_dir_str = onnx_dir.to_str().unwrap_or("onnx_models");
    let script_str = script.to_str().unwrap_or("convert_onnx.py");
    step(
        &format!("🔄 Converting model to ONNX ({})...", dtype.to_uppercase()),
        &[py, script_str, model, onnx_dir_str, dtype],
    )
}
