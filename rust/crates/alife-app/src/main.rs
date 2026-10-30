// Owned by the integration coordinator.

fn main() -> eframe::Result {
    let data_dir = std::path::PathBuf::from("appdata");
    std::fs::create_dir_all(&data_dir).ok();
    alife_ui::run_app(data_dir)
}
