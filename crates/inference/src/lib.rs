use anyhow::{Result, Context};
use tch::Device;

pub mod kv_cache;
pub mod sampling;
pub mod server;
pub mod generator;

// Re-export common types
pub use kv_cache::KVCache;
pub use sampling::{Sampler, SamplingParams};
pub use generator::Generator;

/// Helper function to load model from checkpoint
pub fn load_model(dir: &std::path::Path, device: Device) -> Result<claude_core::ClaudeTransformer> {
    let config_path = dir.join("config.json");
    
    // 1. Load Config
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read model config.json at {:?}", config_path))?;
    let config: claude_core::ModelConfig = serde_json::from_str(&config_str)
        .context("Failed to parse model config.json")?;
        
    // 2. Find latest checkpoint
    let mut checkpoint_path = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut checkpoints: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        
        checkpoints.sort_by_key(|e| e.path());
        checkpoint_path = checkpoints.last().map(|e| e.path());
    }

    // 3. Initialize Model
    let mut vs = tch::nn::VarStore::new(device);
    let model = claude_core::ClaudeTransformer::new(&vs.root(), &config);
    
    if let Some(path) = checkpoint_path {
        println!("Loading weights from {:?}", path);
        claude_core::safetensors_util::load_safetensors(&mut vs, path)
            .context("Failed to load safetensors checkpoint")?;
    } else {
        println!("Warning: No .safetensors checkpoint found in {:?}. Using random weights.", dir);
    }
    
    Ok(model)
}
