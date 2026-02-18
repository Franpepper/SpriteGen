use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct LauncherConfig {
    pub gpu_backend: String,
    pub model: String,
    pub base_resolution: u32,
    pub offline_mode: bool,
    pub hf_token: Option<String>,
    #[serde(default = "default_onnx_dtype")]
    pub onnx_dtype: String,
}

fn default_onnx_dtype() -> String {
    "fp16".into()
}

impl Default for LauncherConfig {
    fn default() -> Self {
        Self {
            gpu_backend: "cpu".into(),
            model: "none".into(),
            base_resolution: 512,
            offline_mode: false,
            hf_token: None,
            onnx_dtype: "fp16".into(),
        }
    }
}

impl LauncherConfig {
    pub fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, json);
        }
    }
}
