fn main() -> eframe::Result {
    let data_dir = std::path::PathBuf::from("appdata");
    std::fs::create_dir_all(&data_dir).ok();
    // crash-log: every panic (incl. wgpu fatal errors sanitized into panics)
    // is appended here so a killed window always leaves an honest trace
    {
        let path = data_dir.join("crash.log");
        std::panic::set_hook(Box::new(move |info| {
            let payload = format!(
                "PANIC {}\n{}\nbacktrace:\n{:?}\n\n",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                info,
                std::backtrace::Backtrace::force_capture(),
            );
            eprintln!("{payload}");
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&path) {
                use std::io::Write as _;
                let _ = f.write_all(payload.as_bytes());
            }
        }));
    }
    alife_ui::run_app(data_dir)
}
