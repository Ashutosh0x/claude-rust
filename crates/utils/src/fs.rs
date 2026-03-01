//! Filesystem utilities.


/// Ensure a directory and all parents exist.
pub fn ensure_dir(path: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

/// Get the size of a file in bytes.
pub fn file_size(path: &str) -> std::io::Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

/// Human-readable file size string.
pub fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    for unit in UNITS {
        if size < 1024.0 {
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PB", size)
}

/// List files matching an extension in a directory.
pub fn list_files_with_ext(dir: &str, ext: &str) -> std::io::Result<Vec<String>> {
    let mut results = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(file_ext) = path.extension().and_then(|e| e.to_str()) {
                if file_ext == ext {
                    results.push(path.to_string_lossy().to_string());
                }
            }
        }
    }
    results.sort();
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_size() {
        assert_eq!(human_size(0), "0.0 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(1_048_576), "1.0 MB");
    }
}
