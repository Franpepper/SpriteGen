mod app;
mod config;
mod gpu;
mod installer;
mod process;
mod python;

fn load_icon() -> eframe::egui::IconData {
    let ico_bytes = include_bytes!("../icon.ico");
    let img = image::load_from_memory_with_format(ico_bytes, image::ImageFormat::Ico)
        .expect("Failed to decode icon")
        .into_rgba8();
    let (w, h) = img.dimensions();
    eframe::egui::IconData {
        rgba: img.into_raw(),
        width: w,
        height: h,
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_title("SpriteGen")
            .with_icon(std::sync::Arc::new(load_icon())),
        ..Default::default()
    };

    eframe::run_native(
        "SpriteGen",
        options,
        Box::new(|cc| Ok(Box::new(app::LauncherApp::new(cc)))),
    )
}
