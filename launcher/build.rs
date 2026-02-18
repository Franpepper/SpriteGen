use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let plugin_src = Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .join("plugin")
        .join("SpriteGen");
    let zip_path = Path::new(&out_dir).join("SpriteGen.zip");

    let _ = std::fs::remove_file(&zip_path);

    let status = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Compress-Archive -Path '{}' -DestinationPath '{}' -Force",
                plugin_src.display(),
                zip_path.display()
            ),
        ])
        .status()
        .expect("Failed to run PowerShell to create plugin zip");

    assert!(status.success(), "Failed to create plugin zip");
    assert!(zip_path.exists(), "Plugin zip not found after creation");

    // Re-run if plugin files change
    // Aseprite plugin
    println!("cargo:rerun-if-changed=../plugin/SpriteGen/");

    // Re-run if server Python files change
    println!("cargo:rerun-if-changed=../server/sd_server.py");
    println!("cargo:rerun-if-changed=../server/backends.py");
    println!("cargo:rerun-if-changed=../server/convert_onnx.py");

    // Embed icon in Windows executable
    println!("cargo:rerun-if-changed=icon.ico");
    let mut res = winres::WindowsResource::new();
    res.set_icon("icon.ico");
    res.compile().expect("Failed to compile Windows resources");
}
