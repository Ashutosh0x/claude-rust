//! Logging setup utilities.

/// Initialize the default logger (env_logger).
/// Set RUST_LOG=info (or debug/trace) to control verbosity.
pub fn init() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
}

/// Initialize with a custom log level.
pub fn init_with_level(level: log::LevelFilter) {
    let _ = env_logger::builder()
        .filter_level(level)
        .try_init();
}
