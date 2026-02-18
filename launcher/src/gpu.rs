use std::process::Command;

pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Both,
    Unknown,
}

impl GpuVendor {
    pub fn label(&self) -> &str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Both => "NVIDIA + AMD",
            Self::Unknown => "Unknown",
        }
    }
}

pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub gpu_names: Vec<String>,
    pub vram_mb: Vec<u64>,
    pub driver_versions: Vec<String>,
}

pub struct SystemInfo {
    pub cpu_name: String,
    pub ram_gb: f64,
}

pub fn detect_gpu() -> GpuInfo {
    let output = Command::new("wmic")
        .args(["path", "win32_videocontroller", "get", "name"])
        .output();

    let names: Vec<String> = match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .lines()
            .skip(1)
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => vec![],
    };

    // VRAM via AdapterRAM (bytes)
    let vram_mb = wmic_query_list(
        &["path", "win32_videocontroller", "get", "AdapterRAM"],
    )
    .iter()
    .filter_map(|s| s.parse::<u64>().ok().map(|b| b / (1024 * 1024)))
    .collect();

    // Driver version
    let driver_versions = wmic_query_list(
        &["path", "win32_videocontroller", "get", "DriverVersion"],
    );

    let text = names.join(" ").to_lowercase();
    let has_nvidia = text.contains("nvidia") || text.contains("geforce");
    let has_amd = text.contains("amd") || text.contains("radeon");
    let has_intel = text.contains("intel");

    let vendor = if has_nvidia && has_amd {
        GpuVendor::Both
    } else if has_nvidia {
        GpuVendor::Nvidia
    } else if has_amd {
        GpuVendor::Amd
    } else if has_intel {
        GpuVendor::Intel
    } else {
        GpuVendor::Unknown
    };

    GpuInfo { vendor, gpu_names: names, vram_mb, driver_versions }
}

pub fn detect_system() -> SystemInfo {
    // CPU name
    let cpu_name = wmic_query_single(&["cpu", "get", "name"])
        .unwrap_or_else(|| "Unknown CPU".into());

    // Total RAM (bytes -> GB)
    let ram_gb = wmic_query_single(&["computersystem", "get", "TotalPhysicalMemory"])
        .and_then(|s| s.parse::<u64>().ok())
        .map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0);

    SystemInfo { cpu_name, ram_gb }
}

// ── WMI helpers ──────────────────────────────────────────────────

fn wmic_query_single(args: &[&str]) -> Option<String> {
    let output = Command::new("wmic").args(args).output().ok()?;
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .map(|l| l.trim().to_string())
        .find(|l| !l.is_empty())
}

fn wmic_query_list(args: &[&str]) -> Vec<String> {
    let output = match Command::new("wmic").args(args).output() {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}
