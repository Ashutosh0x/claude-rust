//! Configuration loading helpers.

use std::path::Path;

/// Load a YAML configuration file and deserialize it.
pub fn load_yaml<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    let content = std::fs::read_to_string(path)?;
    let config: T = serde_yaml::from_str(&content)?;
    Ok(config)
}

/// Load a JSON configuration file.
pub fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    let content = std::fs::read_to_string(path)?;
    let config: T = serde_json::from_str(&content)?;
    Ok(config)
}

/// Save a configuration to YAML.
pub fn save_yaml<T: serde::Serialize>(path: &str, config: &T) -> anyhow::Result<()> {
    let content = serde_yaml::to_string(config)?;
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}

/// Save a configuration to JSON.
pub fn save_json<T: serde::Serialize>(path: &str, config: &T) -> anyhow::Result<()> {
    let content = serde_json::to_string_pretty(config)?;
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}
