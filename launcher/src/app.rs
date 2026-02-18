use eframe::egui;
use std::path::PathBuf;
use eframe::egui::Button;

const PLUGIN_ZIP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/SpriteGen.zip"));

const SERVER_SD_SERVER: &str = include_str!("../../server/sd_server.py");
const SERVER_BACKENDS: &str = include_str!("../../server/backends.py");
const SERVER_CONVERT_ONNX: &str = include_str!("../../server/convert_onnx.py");

use crate::config::LauncherConfig;
use crate::gpu::{self, GpuInfo, GpuVendor, SystemInfo};
use crate::installer::{self, InstallStep};
use crate::process::{self, ProcessHandle};
use crate::python::{self, PythonInfo};

const MODELS: &[(&str, &str, bool)] = &[
    ("stabilityai/stable-diffusion-3.5-large-turbo", "SD 3.5 Large Turbo (~14-18GB)", true),
    ("stabilityai/stable-diffusion-3.5-large", "SD 3.5 Large (~14-18GB)", true),
    ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL Base 1.0 (~7GB)", false),
    ("runwayml/stable-diffusion-v1-5", "SD v1.5 (~5GB)", false),
    ("none", "None (load later)", false),
];

const RESOLUTIONS: &[(u32, &str)] = &[
    (128, "128x128 (Fastest)"),
    (256, "256x256 (Fast)"),
    (512, "512x512 (Balanced)"),
    (768, "768x768 (Quality)"),
    (1024, "1024x1024 (Max Quality)"),
];

const BACKENDS: &[(&str, &str)] = &[
    ("cuda", "CUDA (NVIDIA)"),
    ("rocm", "ROCm (AMD)"),
    ("windowsml", "Windows ML"),
    ("cpu", "CPU"),
];

const MAX_LOG_LINES: usize = 10_000;

#[derive(PartialEq)]
enum AppStatus {
    Idle,
    Installing,
    ServerRunning,
    Error(String),
}

pub struct LauncherApp {
    // System info (detected once at startup)
    python_info: Option<PythonInfo>,
    gpu_info: GpuInfo,
    system_info: SystemInfo,
    installed_backend: Option<String>,
    project_dir: PathBuf,
    config_path: PathBuf,

    // User config
    config: LauncherConfig,
    last_saved_config: LauncherConfig,
    hf_token_input: String,
    hf_authenticated: bool,

    // Runtime state
    status: AppStatus,
    log_lines: Vec<String>,
    current_process: Option<ProcessHandle>,
    install_queue: Vec<InstallStep>,
    install_step_idx: usize,

    // ONNX conversion popup
    show_onnx_popup: bool,
}

impl LauncherApp {
    pub fn new(_cc: &eframe::CreationContext) -> Self {
        let project_dir = find_project_dir();
        ensure_server_files(&project_dir);
        let config_path = project_dir.join("launcher_config.json");
        let config = LauncherConfig::load(&config_path);

        let python_info = python::find_python();
        let gpu_info = gpu::detect_gpu();
        let system_info = gpu::detect_system();

        let venv_py = python::venv_python(&project_dir);
        let installed_backend = python::get_installed_backend(&venv_py);

        let hf_token_input = config.hf_token.clone().unwrap_or_default();
        let hf_authenticated = config.hf_token.as_ref().map_or(false, |t| !t.is_empty());

        let mut app = Self {
            python_info,
            gpu_info,
            system_info,
            installed_backend,
            project_dir,
            config_path,
            last_saved_config: config.clone(),
            config,
            hf_token_input,
            hf_authenticated,
            status: AppStatus::Idle,
            log_lines: Vec::new(),
            current_process: None,
            install_queue: Vec::new(),
            install_step_idx: 0,
            show_onnx_popup: false,
        };


        // Auto-select recommended backend if still on default "cpu"
        if app.config.gpu_backend == "cpu" {
            app.config.gpu_backend = app.recommended_backend().to_string();
        }

        app
    }

    fn recommended_backend(&self) -> &str {
        match &self.gpu_info.vendor {
            GpuVendor::Nvidia | GpuVendor::Both => "cuda",
            GpuVendor::Amd => "rocm",
            GpuVendor::Intel => "windowsml",
            GpuVendor::Unknown => "cpu",
        }
    }

    fn is_busy(&self) -> bool {
        matches!(self.status, AppStatus::Installing)
    }

    fn venv_python(&self) -> PathBuf {
        python::venv_python(&self.project_dir)
    }

    // Install flow

    fn start_install(&mut self) {
        let py_info = match &self.python_info {
            Some(p) => p,
            None => {
                self.status = AppStatus::Error("Python not found".into());
                return;
            }
        };

        if py_info.major != 3 || py_info.minor != 12 {
            self.status = AppStatus::Error(format!(
                "Python 3.12 required (found {}.{}). Install Python 3.12 from python.org.",
                py_info.major, py_info.minor
            ));
            return;
        }

        let mut steps: Vec<InstallStep> = Vec::new();

        let backend_changed = self
            .installed_backend
            .as_deref()
            .map_or(false, |ib| ib != self.config.gpu_backend);

        // Delete venv when backend changes
        if backend_changed && python::venv_exists(&self.project_dir) {
            self.log_lines.push(format!(
                "Backend changed ({} -> {}), recreating virtual environment...",
                self.installed_backend
                    .as_deref()
                    .unwrap_or("?")
                    .to_uppercase(),
                self.config.gpu_backend.to_uppercase()
            ));
            let venv_dir = self
                .project_dir
                .join("venv")
                .to_str()
                .unwrap_or("venv")
                .to_string();
            steps.push(InstallStep {
                label: "🗑️ Removing old virtual environment...".into(),
                args: vec![
                    "cmd.exe".into(),
                    "/c".into(),
                    "rmdir".into(),
                    "/s".into(),
                    "/q".into(),
                    venv_dir,
                ],
                allow_failure: false,
            });
        }

        // Create venv if needed (use system python for this step)
        if !python::venv_exists(&self.project_dir) || backend_changed {
            let sys_py = py_info.path.to_str().unwrap_or("python").to_string();
            steps.push(InstallStep {
                label: "📦 Creating virtual environment...".into(),
                args: vec![sys_py, "-m".into(), "venv".into(), "venv".into()],
                allow_failure: false,
            });
        }

        // Pip install steps (using venv python)
        let venv_py = self.venv_python();
        let fresh_venv = !python::venv_exists(&self.project_dir) || backend_changed;
        steps.extend(installer::build_install_steps(&venv_py, &self.config.gpu_backend, fresh_venv));

        self.install_queue = steps;
        self.install_step_idx = 0;
        self.status = AppStatus::Installing;
        self.log_lines.push("─── Starting installation ───".into());
        self.launch_current_install_step();
    }

    fn launch_current_install_step(&mut self) {
        if self.install_step_idx >= self.install_queue.len() {
            self.finish_install();
            return;
        }

        let step = &self.install_queue[self.install_step_idx];
        self.log_lines.push(format!("> {}", step.label));

        match process::spawn_process(&step.args, &self.project_dir) {
            Ok(handle) => {
                self.current_process = Some(handle);
            }
            Err(e) => {
                self.log_lines.push(format!("❌ Error: {}", e));
                self.status = AppStatus::Error(e);
                self.install_queue.clear();
            }
        }
    }

    fn finish_install(&mut self) {
        self.status = AppStatus::Idle;
        self.log_lines.push("✅ Installation complete!".into());
        self.install_queue.clear();
        // Re-detect installed backend
        self.installed_backend = python::get_installed_backend(&self.venv_python());

        // Remind WindowsML users to convert models
        if self.config.gpu_backend == "windowsml" {
            self.show_onnx_popup = true;
        }
    }

    // ── Server flow ────────────────────────────────────────────────

    fn start_server(&mut self) {
        // WindowsML pre-check: ensure ONNX model exists before starting server
        if self.config.gpu_backend == "windowsml" && self.config.model != "none" {
            if !self.is_model_converted() {
                self.log_lines.push(format!(
                    "❌ ONNX model not found for '{}'. Use the 'Convert to ONNX' button first.",
                    self.config.model
                ));
                self.status = AppStatus::Error(
                    "Model not converted to ONNX. Use 'Convert to ONNX' button.".into(),
                );
                return;
            }
        }

        let python = self.venv_python();
        let py_str = python.to_str().unwrap_or("python").to_string();

        let model_arg = if self.config.model == "none" {
            "None".to_string()
        } else {
            format!("'{}'", self.config.model)
        };
        let offline_arg = if self.config.offline_mode { "True" } else { "False" };

        let server_dir = self
            .project_dir
            .join("server")
            .to_str()
            .unwrap_or("server")
            .replace('\\', "/");

        let script = format!(
            "import sys; sys.path.insert(0, '{}'); from sd_server import main; main(default_model_to_load={}, offline={}, base_resolution={})",
            server_dir, model_arg, offline_arg, self.config.base_resolution
        );

        self.log_lines.push("🚀 Starting server...".into());

        let args = vec![py_str, "-u".into(), "-c".into(), script];
        match process::spawn_process(&args, &self.project_dir) {
            Ok(handle) => {
                self.current_process = Some(handle);
                self.status = AppStatus::ServerRunning;
            }
            Err(e) => {
                self.log_lines.push(format!("❌ Error starting server: {}", e));
                self.status = AppStatus::Error(e);
            }
        }
    }

    fn stop_server(&mut self) {
        if let Some(proc) = &mut self.current_process {
            proc.kill();
        }
        self.current_process = None;
        self.status = AppStatus::Idle;
        self.log_lines.push("⏹ Server stopped.".into());
    }

    // ── ONNX conversion ─────────────────────────────────────────────

    fn is_model_converted(&self) -> bool {
        let escaped = self.config.model.replace('/', "--");
        let index = self.project_dir
            .join("onnx_models")
            .join(&escaped)
            .join("model_index.json");
        index.exists()
    }

    fn invalidate_onnx_conversion(&mut self) {
        let escaped = self.config.model.replace('/', "--");
        let onnx_dir = self.project_dir.join("onnx_models").join(&escaped);
        if onnx_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&onnx_dir) {
                self.log_lines.push(format!(
                    "⚠ Failed to remove ONNX cache at '{}': {}",
                    onnx_dir.display(),
                    e
                ));
            }
        }
    }

    fn start_onnx_conversion(&mut self) {
        let venv_py = self.venv_python();
        if !venv_py.exists() {
            self.log_lines.push(
                "⚠ Install dependencies first before converting models.".into(),
            );
            return;
        }

        let step = installer::build_onnx_convert_step(
            &venv_py,
            &self.config.model,
            &self.project_dir,
            &self.config.onnx_dtype,
        );
        self.install_queue = vec![step];
        self.install_step_idx = 0;
        self.status = AppStatus::Installing;
        self.log_lines
            .push("─── Starting ONNX conversion ───".into());
        self.launch_current_install_step();
    }

    // ── HuggingFace ────────────────────────────────────────────────

    fn login_hf(&mut self) {
        let token = self.hf_token_input.trim().to_string();
        if token.is_empty() {
            return;
        }

        let venv_py = self.venv_python();
        if !venv_py.exists() {
            self.log_lines
                .push("⚠ Install dependencies first before logging in to HuggingFace.".into());
            return;
        }

        let script = format!(
            "from huggingface_hub import login; login(token='{}')",
            token.replace('\'', "\\'")
        );

        self.log_lines.push("Logging in to HuggingFace...".into());

        let output = std::process::Command::new(venv_py.to_str().unwrap_or("python"))
            .args(["-c", &script])
            .current_dir(&self.project_dir)
            .output();

        match output {
            Ok(o) if o.status.success() => {
                self.hf_authenticated = true;
                self.config.hf_token = Some(token);
                self.log_lines.push("✅ HuggingFace login successful.".into());
            }
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr);
                self.log_lines
                    .push(format!("❌ HuggingFace login failed: {}", err.trim()));
            }
            Err(e) => {
                self.log_lines.push(format!("❌ Error: {}", e));
            }
        }
    }

    // ── Plugin install ─────────────────────────────────────────

    fn install_plugin(&mut self) {
        let ext_path = self.project_dir.join("SpriteGen.aseprite-extension");

        self.log_lines
            .push("📦 Installing SpriteGen plugin...".into());

        match std::fs::write(&ext_path, PLUGIN_ZIP) {
            Ok(_) => {
                self.log_lines
                    .push("✅ Created SpriteGen.aseprite-extension".into());
                let ext_str = ext_path.to_str().unwrap_or("");
                let _ = std::process::Command::new("cmd")
                    .args(["/c", "start", "", ext_str])
                    .spawn();
                self.log_lines
                    .push("📦 Opening Aseprite extension installer...".into());
            }
            Err(e) => {
                self.log_lines
                    .push(format!("❌ Failed to write extension: {}", e));
            }
        }
    }

    // ── Config persistence ─────────────────────────────────────────

    fn save_config_if_changed(&mut self) {
        if self.config != self.last_saved_config {
            self.config.save(&self.config_path);
            self.last_saved_config = self.config.clone();
        }
    }

    // ── Process polling ────────────────────────────────────────────

    fn poll_process(&mut self) {
        let (lines, finished, success) = match &mut self.current_process {
            Some(proc) => proc.poll(),
            None => return,
        };

        self.log_lines.extend(lines);

        if !finished {
            return;
        }

        self.current_process = None;

        match &self.status {
            AppStatus::Installing => {
                let allow_failure = self
                    .install_queue
                    .get(self.install_step_idx)
                    .map_or(false, |s| s.allow_failure);

                if success || allow_failure {
                    self.install_step_idx += 1;
                    self.launch_current_install_step();
                } else {
                    let step_label = self
                        .install_queue
                        .get(self.install_step_idx)
                        .map(|s| s.label.clone())
                        .unwrap_or_default();
                    self.log_lines.push(format!("❌ Step failed: {}", step_label));
                    self.status = AppStatus::Error(format!("Installation failed at: {}", step_label));
                    self.install_queue.clear();
                }
            }
            AppStatus::ServerRunning => {
                self.log_lines.push("Server process ended.".into());
                self.status = AppStatus::Idle;
            }
            _ => {
                self.status = AppStatus::Idle;
            }
        }
    }

    // ── UI drawing ─────────────────────────────────────────────────

    fn draw_system_info(&self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("SYSTEM INFO").strong());
        egui::Grid::new("system_info")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                // CPU
                ui.label("CPU:");
                ui.label(format!(
                    "{}",
                    self.system_info.cpu_name
                ));
                ui.end_row();

                // RAM
                ui.label("RAM:");
                ui.label(format!("{:.1} GB", self.system_info.ram_gb));
                ui.end_row();

                // GPU(s) with VRAM and driver
                ui.label("GPU:");
                if self.gpu_info.gpu_names.is_empty() {
                    ui.label("None detected");
                } else {
                    ui.vertical(|ui| {
                        for (i, name) in self.gpu_info.gpu_names.iter().enumerate() {
                            let mut parts = vec![name.clone()];
                            if let Some(vram) = self.gpu_info.vram_mb.get(i) {
                                if *vram > 0 {
                                    parts.push(format!("{} MB VRAM", vram));
                                }
                            }
                            if let Some(drv) = self.gpu_info.driver_versions.get(i) {
                                if !drv.is_empty() {
                                    parts.push(format!("Driver {}", drv));
                                }
                            }
                            ui.label(parts.join(" | "));
                        }
                        ui.label(format!("Vendor: {}", self.gpu_info.vendor.label()));
                    });
                }
                ui.end_row();

                // Python
                ui.label("Python:");
                if let Some(py) = &self.python_info {
                    if py.major == 3 && py.minor == 12 {
                        ui.label(format!("{} ({})", py.version, py.path.display()));
                    } else {
                        ui.colored_label(
                            egui::Color32::RED,
                            format!("{} - Python 3.12 required!", py.version),
                        );
                    }
                } else {
                    ui.colored_label(
                        egui::Color32::RED,
                        "Not found! Install Python 3.12 from python.org",
                    );
                }
                ui.end_row();

                // Installed Backend
                ui.label("Backend:");
                if let Some(backend) = &self.installed_backend {
                    ui.label(backend.to_uppercase());
                } else {
                    ui.colored_label(egui::Color32::YELLOW, "Not installed");
                }
                ui.end_row();

                // Status
                ui.label("Status:");
                match &self.status {
                    AppStatus::Idle => {
                        ui.label("Idle");
                    }
                    AppStatus::Installing => {
                        let progress = format!(
                            "Installing ({}/{})",
                            self.install_step_idx + 1,
                            self.install_queue.len()
                        );
                        ui.colored_label(egui::Color32::YELLOW, progress);
                    }
                    AppStatus::ServerRunning => {
                        ui.colored_label(egui::Color32::GREEN, "● Server Running");
                    }
                    AppStatus::Error(e) => {
                        ui.colored_label(egui::Color32::RED, format!("Error: {}", e));
                    }
                };
                ui.end_row();
            });
    }

    fn draw_configuration(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("CONFIGURATION").strong());
        let enabled = !self.is_busy() && !matches!(self.status, AppStatus::ServerRunning);

        egui::Grid::new("config_grid")
            .num_columns(2)
            .spacing([20.0, 6.0])
            .show(ui, |ui| {
                // Backend selector
                ui.label("Backend:");
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(enabled, |ui| {
                        egui::ComboBox::from_id_salt("backend_combo")
                            .selected_text(
                                BACKENDS
                                    .iter()
                                    .find(|(id, _)| *id == self.config.gpu_backend)
                                    .map(|(_, label)| *label)
                                    .unwrap_or("Unknown"),
                            )
                            .show_ui(ui, |ui| {
                                for &(id, label) in BACKENDS {
                                    ui.selectable_value(
                                        &mut self.config.gpu_backend,
                                        id.to_string(),
                                        label,
                                    );
                                }
                            });
                    });
                    if self.config.gpu_backend == "windowsml" {
                        ui.colored_label(egui::Color32::YELLOW, "⚠ Experimental");
                    }
                });
                ui.end_row();

                // Model selector
                ui.label("Model:");
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(enabled, |ui| {
                        egui::ComboBox::from_id_salt("model_combo")
                            .selected_text(
                                MODELS
                                    .iter()
                                    .find(|(id, _, _)| *id == self.config.model)
                                    .map(|(_, label, _)| *label)
                                    .unwrap_or("Unknown"),
                            )
                            .show_ui(ui, |ui| {
                                for &(id, label, gated) in MODELS {
                                    if gated && !self.hf_authenticated {
                                        ui.add_enabled(
                                            false,
                                            Button::selectable(
                                                false,
                                                format!("{} (requires HF login)", label),
                                            ),
                                        );
                                    } else {
                                        ui.selectable_value(
                                            &mut self.config.model,
                                            id.to_string(),
                                            label,
                                        );
                                    }
                                }
                            });
                    });

                    // ONNX conversion controls — only for WindowsML with a real model
                    if self.config.gpu_backend == "windowsml" && self.config.model != "none" {
                        let already_converted = self.is_model_converted();
                        let can_act = enabled && self.installed_backend.is_some();

                        // Convert / Reconvert button
                        let btn_label = if already_converted {
                            "🔄 Reconvert"
                        } else {
                            "🔄 Convert to ONNX"
                        };
                        if ui.add_enabled(can_act, egui::Button::new(btn_label)).clicked() {
                            if already_converted {
                                self.invalidate_onnx_conversion();
                            }
                            self.start_onnx_conversion();
                        }

                        // Clean cache button (only if converted)
                        if already_converted {
                            if ui.add_enabled(can_act, egui::Button::new("🗑️")).on_hover_text("Delete ONNX cache for this model").clicked() {
                                self.invalidate_onnx_conversion();
                                self.log_lines.push("🗑️ ONNX cache cleaned for this model.".into());
                            }
                            ui.colored_label(egui::Color32::GREEN, "✓ Converted");
                        }
                    }
                });
                ui.end_row();

                // ONNX precision selector — only for WindowsML
                if self.config.gpu_backend == "windowsml" {
                    ui.label("ONNX Precision:");
                    ui.add_enabled_ui(enabled, |ui| {
                        egui::ComboBox::from_id_salt("onnx_dtype_combo")
                            .selected_text(match self.config.onnx_dtype.as_str() {
                                "fp16" => "FP16 (Recommended)",
                                "bf16" => "BF16",
                                "fp32" => "FP32 (High memory)",
                                _ => "FP16 (Recommended)",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.config.onnx_dtype,
                                    "fp16".to_string(),
                                    "FP16 (Recommended)",
                                );
                                ui.selectable_value(
                                    &mut self.config.onnx_dtype,
                                    "bf16".to_string(),
                                    "BF16",
                                );
                                ui.selectable_value(
                                    &mut self.config.onnx_dtype,
                                    "fp32".to_string(),
                                    "FP32 (High memory)",
                                );
                            });
                    });
                    ui.end_row();
                }

                // Resolution selector
                ui.label("Resolution:");
                ui.add_enabled_ui(enabled, |ui| {
                    egui::ComboBox::from_id_salt("resolution_combo")
                        .selected_text(
                            RESOLUTIONS
                                .iter()
                                .find(|(val, _)| *val == self.config.base_resolution)
                                .map(|(_, label)| *label)
                                .unwrap_or("512x512 (Fast)"),
                        )
                        .show_ui(ui, |ui| {
                            for &(val, label) in RESOLUTIONS {
                                ui.selectable_value(
                                    &mut self.config.base_resolution,
                                    val,
                                    label,
                                );
                            }
                        });
                });
                ui.end_row();

                // Offline mode
                ui.label("Offline Mode:");
                ui.add_enabled(
                    enabled,
                    egui::Checkbox::new(&mut self.config.offline_mode, "Use only cached models"),
                );
                ui.end_row();

                // HuggingFace token
                ui.label("HuggingFace:");
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        enabled,
                        egui::TextEdit::singleline(&mut self.hf_token_input)
                            .hint_text("Token (hf_...)")
                            .password(true)
                            .desired_width(250.0),
                    );
                    if ui
                        .add_enabled(
                            enabled && !self.hf_token_input.trim().is_empty(),
                            egui::Button::new("Login"),
                        )
                        .clicked()
                    {
                        self.login_hf();
                    }
                    if self.hf_authenticated {
                        ui.colored_label(egui::Color32::GREEN, "✓ Logged in");
                    }
                });
                ui.end_row();
            });
    }

    fn draw_actions(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let has_python = self.python_info.is_some();

            // Install button
            let can_install =
                has_python && !self.is_busy() && !matches!(self.status, AppStatus::ServerRunning);
            if ui
                .add_enabled(can_install, egui::Button::new("📦 Install Dependencies"))
                .clicked()
            {
                self.start_install();
            }

            ui.add_space(8.0);

            // Start button
            let can_start = has_python
                && self.installed_backend.is_some()
                && !self.is_busy()
                && !matches!(self.status, AppStatus::ServerRunning);
            if ui
                .add_enabled(can_start, egui::Button::new("▶ Start Server"))
                .clicked()
            {
                self.start_server();
            }

            // Stop button
            let can_stop = matches!(self.status, AppStatus::ServerRunning);
            if ui
                .add_enabled(can_stop, egui::Button::new("⏹ Stop Server"))
                .clicked()
            {
                self.stop_server();
            }

            ui.add_space(8.0);

            // Install Plugin button
            let can_install_plugin =
                !self.is_busy() && !matches!(self.status, AppStatus::ServerRunning);
            if ui
                .add_enabled(can_install_plugin, egui::Button::new("🔌 Install Plugin"))
                .clicked()
            {
                self.install_plugin();
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Clear Log").clicked() {
                    self.log_lines.clear();
                }
            });
        });
    }

    fn draw_log(&self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("LOG OUTPUT").strong());
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &self.log_lines {
                    ui.label(egui::RichText::new(line).monospace().size(12.0));
                }
            });
    }
}

fn ensure_server_files(base_dir: &std::path::Path) {
    let server_dir = base_dir.join("server");
    let _ = std::fs::create_dir_all(&server_dir);

    let files: &[(&str, &str)] = &[
        ("sd_server.py", SERVER_SD_SERVER),
        ("backends.py", SERVER_BACKENDS),
        ("convert_onnx.py", SERVER_CONVERT_ONNX),
    ];

    for (name, content) in files {
        let path = server_dir.join(name);
        let _ = std::fs::write(&path, content);
    }
}

fn find_project_dir() -> PathBuf {
    let marker = std::path::Path::new("server").join("sd_server.py");

    // Dev mode: check CWD
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join(&marker).exists() {
        return cwd;
    }

    // Dev mode: walk up from exe location
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        while let Some(d) = dir {
            if d.join(&marker).exists() {
                return d;
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }

        // Distribution mode: use exe's parent directory
        if let Some(exe_dir) = exe.parent() {
            return exe_dir.to_path_buf();
        }
    }

    cwd
}

impl eframe::App for LauncherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll running process
        self.poll_process();

        if self.current_process.is_some() {
            ctx.request_repaint();
        }

        // Trim log if too long
        if self.log_lines.len() > MAX_LOG_LINES {
            let drain = self.log_lines.len() - MAX_LOG_LINES;
            self.log_lines.drain(0..drain);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Title bar
            ui.horizontal(|ui| {
                ui.heading("SpriteGen");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("v0.2.0");
                });
            });
            ui.separator();

            // System Info
            self.draw_system_info(ui);
            ui.separator();

            // Configuration
            self.draw_configuration(ui);
            ui.separator();

            // Action buttons
            self.draw_actions(ui);
            ui.separator();

            // Log output (fills remaining space)
            self.draw_log(ui);
        });

        // ONNX conversion reminder popup
        if self.show_onnx_popup {
            egui::Window::new("ONNX Conversion Required")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(
                        "Windows ML requires models to be converted to ONNX format \
                         before starting the server.",
                    );
                    ui.add_space(4.0);
                    ui.label(
                        "Use the 'Convert to ONNX' button next to the model selector \
                         to convert your chosen model.",
                    );
                    ui.add_space(8.0);
                    if ui.button("OK").clicked() {
                        self.show_onnx_popup = false;
                    }
                });
        }

        // Save config when it changes
        self.save_config_if_changed();
    }
}

impl Drop for LauncherApp {
    fn drop(&mut self) {
        if let Some(proc) = &mut self.current_process {
            proc.kill();
        }
        self.config.save(&self.config_path);
    }
}
