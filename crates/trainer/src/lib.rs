pub mod dataset;
pub mod train;
pub mod checkpoint;
pub mod data_loader;
pub mod metrics;
pub mod optim;
pub mod schedule;
pub mod train_loop;

pub use train::Trainer;
pub use checkpoint::CheckpointManager;
pub use data_loader::DataLoader;
pub use metrics::{MetricsLogger, StepMetrics};
pub use schedule::CosineScheduler;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub context_length: usize,
    pub epochs: usize,
    pub save_every: usize,
    pub checkpoint_dir: String,
    pub warmup_steps: Option<usize>,
    pub weight_decay: Option<f64>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 4,
            context_length: 128,
            epochs: 1,
            save_every: 100,
            checkpoint_dir: "./checkpoints".to_string(),
            warmup_steps: Some(0),
            weight_decay: Some(0.01),
        }
    }
}
