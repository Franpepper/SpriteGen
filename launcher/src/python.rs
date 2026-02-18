use std::path::{Path, PathBuf};
use std::process::Command;

pub struct PythonInfo {
    pub path: PathBuf,
    pub version: String,
    pub major: u32,
    pub minor: u32,
}

pub fn find_python() -> Option<PythonInfo> {
    for cmd in &["py", "python"] {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                // Python may print version to stdout or stderr
                let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let raw = if raw.is_empty() {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                } else {
                    raw
                };

                if let Some(ver) = raw.strip_prefix("Python ") {
                    let parts: Vec<&str> = ver.split('.').collect();
                    let major: u32 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let minor: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

                    if major == 3 && minor == 12 {
                        return Some(PythonInfo {
                            path: PathBuf::from(cmd),
                            version: ver.to_string(),
                            major,
                            minor,
                        });
                    }
                }
            }
        }
    }
    None
}

pub fn venv_python(project_dir: &Path) -> PathBuf {
    project_dir.join("venv").join("Scripts").join("python.exe")
}

pub fn venv_exists(project_dir: &Path) -> bool {
    venv_python(project_dir).exists()
}

pub fn get_installed_backend(venv_py: &Path) -> Option<String> {
    if !venv_py.exists() {
        return None;
    }

    let script = concat!(
        "try:\n",
        "    import onnxruntime as ort\n",
        "    if 'DmlExecutionProvider' in ort.get_available_providers():\n",
        "        print('windowsml')\n",
        "    else:\n",
        "        raise ImportError\n",
        "except (ImportError, Exception):\n",
        "    try:\n",
        "        import torch\n",
        "        hip = getattr(torch.version, 'hip', None)\n",
        "        if hip:\n",
        "            print('rocm')\n",
        "        elif torch.version.cuda:\n",
        "            print('cuda')\n",
        "        else:\n",
        "            print('cpu')\n",
        "    except ImportError:\n",
        "        print('none')\n",
    );

    let output = Command::new(venv_py).args(["-c", script]).output().ok()?;
    let result = String::from_utf8_lossy(&output.stdout).trim().to_string();

    if result == "none" || result.is_empty() {
        None
    } else {
        Some(result)
    }
}
