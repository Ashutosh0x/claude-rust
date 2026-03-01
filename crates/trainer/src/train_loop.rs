/// High-level training loop that combines all trainer components.
///
/// Wires together: DataLoader, Optimizer, Scheduler, Metrics, and Checkpointing.

use anyhow::Result;
use tch::{nn, nn::OptimizerConfig, Device};

use claude_core::{ClaudeTransformer, ModelConfig};
use crate::checkpoint::CheckpointManager;
use crate::data_loader::DataLoader;
use crate::metrics::{MetricsLogger, StepMetrics};
use crate::schedule::{CosineScheduler, apply_lr};
use crate::TrainerConfig;

/// Full-featured training loop.
pub fn train_loop(
    model_config: &ModelConfig,
    trainer_config: &TrainerConfig,
    data_path: &str,
    device: Device,
) -> Result<()> {
    // 1. Initialize model
    let vs = nn::VarStore::new(device);
    let model = ClaudeTransformer::new(&vs.root(), model_config);

    // 2. Initialize optimizer
    let mut optimizer = nn::AdamW::default()
        .build(&vs, trainer_config.learning_rate)?;

    // 3. Setup checkpoint manager
    let ckpt = CheckpointManager::new(&trainer_config.checkpoint_dir);

    // 4. Setup scheduler
    let total_steps = trainer_config.epochs * 1000;
    let scheduler = CosineScheduler::new(
        trainer_config.learning_rate,
        trainer_config.learning_rate * 0.1,
        trainer_config.warmup_steps.unwrap_or(0),
        total_steps,
    );

    // 5. Setup metrics logger
    let metrics = MetricsLogger::new(Some("metrics.jsonl".to_string()));

    // 6. Setup data loader
    let mut loader = DataLoader::from_bin_file(
        data_path,
        trainer_config.context_length,
        trainer_config.batch_size,
        device,
    )?;

    println!("Starting training: {} epochs, {} batches available",
        trainer_config.epochs, loader.num_batches());

    // 7. Training loop
    let mut global_step = 0usize;
    let mut last_loss = 0.0f64;

    for epoch in 0..trainer_config.epochs {
        let mut epoch_loss = 0.0;
        let batches_per_epoch = loader.num_batches().max(1);
        loader.reset();

        for _batch_idx in 0..batches_per_epoch {
            // Get LR from scheduler
            let lr = scheduler.get_lr(global_step);
            apply_lr(&mut optimizer, lr);

            // Get batch
            let (input, target) = loader.next_batch();

            // Forward
            let logits = model.forward(&input, None);
            let (b, t, v) = logits.size3()?;
            let logits_flat = logits.view([b * t, v]);
            let target_flat = target.view([b * t]);
            let loss = logits_flat.cross_entropy_for_logits(&target_flat);

            // Backward + step
            optimizer.backward_step(&loss);

            let loss_val = loss.double_value(&[]);
            epoch_loss += loss_val;
            last_loss = loss_val;
            global_step += 1;

            // Log every 10 steps
            if global_step % 10 == 0 {
                let step_metrics = StepMetrics::new(global_step, epoch, loss_val, lr);
                metrics.log(&step_metrics);
            }

            // Checkpoint
            if global_step % trainer_config.save_every == 0 {
                ckpt.save(&vs, model_config, global_step, loss_val)?;
            }
        }

        let avg_loss = epoch_loss / batches_per_epoch as f64;
        metrics.log_epoch_summary(epoch, avg_loss, batches_per_epoch);
    }

    // Final checkpoint
    ckpt.save(&vs, model_config, global_step, last_loss)?;

    println!("Training complete! Total steps: {}", global_step);
    Ok(())
}
