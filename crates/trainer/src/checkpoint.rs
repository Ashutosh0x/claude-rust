use anyhow::{Result, Context};
use std::path::PathBuf;
use tch::nn;

/// Manages saving/loading model checkpoints in safetensors format.
pub struct CheckpointManager {
    pub checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: dir.into(),
        }
    }

    /// Ensure the checkpoint directory exists.
    pub fn ensure_dir(&self) -> Result<()> {
        if !self.checkpoint_dir.exists() {
            std::fs::create_dir_all(&self.checkpoint_dir)
                .with_context(|| format!("Failed to create checkpoint dir: {:?}", self.checkpoint_dir))?;
        }
        Ok(())
    }

    /// Save model weights and config.
    pub fn save(
        &self,
        vs: &nn::VarStore,
        config: &claude_core::ModelConfig,
        step: usize,
        loss: f64,
    ) -> Result<PathBuf> {
        self.ensure_dir()?;

        let weights_path = self.checkpoint_dir.join(format!("checkpoint_step_{}.safetensors", step));
        vs.save(&weights_path)
            .with_context(|| format!("Failed to save weights to {:?}", weights_path))?;

        let config_path = self.checkpoint_dir.join("config.json");
        let config_json = serde_json::to_string_pretty(config)?;
        std::fs::write(&config_path, config_json)?;

        let meta = serde_json::json!({
            "step": step,
            "loss": loss,
        });
        let meta_path = self.checkpoint_dir.join("training_meta.json");
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

        println!("[Checkpoint] Saved step {} (loss={:.4}) to {:?}", step, loss, weights_path);
        Ok(weights_path)
    }

    /// Load the latest checkpoint from the directory.
    pub fn load_latest(&self, vs: &mut nn::VarStore) -> Result<Option<usize>> {
        if !self.checkpoint_dir.exists() {
            return Ok(None);
        }

        let mut checkpoints: Vec<_> = std::fs::read_dir(&self.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.starts_with("checkpoint_step_") && n.ends_with(".safetensors"))
            })
            .collect();

        if checkpoints.is_empty() {
            return Ok(None);
        }

        checkpoints.sort_by_key(|e| e.path());
        let latest = checkpoints.last().unwrap().path();

        let step = latest
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("checkpoint_step_"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        vs.load(&latest)
            .with_context(|| format!("Failed to load checkpoint: {:?}", latest))?;

        println!("[Checkpoint] Resumed from step {} ({:?})", step, latest);
        Ok(Some(step))
    }

    /// List all available checkpoints sorted by step.
    pub fn list_checkpoints(&self) -> Vec<(usize, PathBuf)> {
        if !self.checkpoint_dir.exists() {
            return Vec::new();
        }

        let mut results: Vec<(usize, PathBuf)> = std::fs::read_dir(&self.checkpoint_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                let step = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("checkpoint_step_"))
                    .and_then(|s| s.parse::<usize>().ok())?;
                Some((step, path))
            })
            .collect();

        results.sort_by_key(|(step, _)| *step);
        results
    }
}
