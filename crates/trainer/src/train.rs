use anyhow::Result;
use std::path::PathBuf;
use tch::{nn, nn::OptimizerConfig, Device};

use claude_core::{ClaudeTransformer, ModelConfig};
use tokenizer::BPE;

use crate::dataset::TextDataset;
use crate::TrainerConfig;

pub struct Trainer {
    config: TrainerConfig,
    model: ClaudeTransformer,
    optimizer: nn::Optimizer,
    device: Device,
    vs: nn::VarStore,
}

impl Trainer {
    pub fn new(
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        device: Device,
    ) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let model = ClaudeTransformer::new(&vs.root(), &model_config);
        
        let optimizer = nn::AdamW::default()
            .build(&vs, trainer_config.learning_rate)?;

        Ok(Self {
            config: trainer_config,
            model,
            optimizer,
            device,
            vs,
        })
    }

    pub fn train(&mut self, text: &str, tokenizer: &BPE) -> Result<()> {
        let dataset = TextDataset::new(text, tokenizer, self.config.context_length, self.device);
        
        println!("Starting training with configuration: {:?}", self.config);
        
        for epoch in 0..self.config.epochs {
            // Training Loop
            let mut epoch_loss = 0.0;
            let num_batches = 100; // Define batches per epoch or iterate fully
            
            for batch_idx in 0..num_batches {
                let (input, target) = dataset.sample_batch(self.config.batch_size);
                
                // Forward pass
                // Returns logits
                let logits = self.model.forward(&input, None);
                
                // Reshape for loss: [B*T, V] vs [B*T]
                let (b, t, v) = logits.size3()?;
                let logits_flat = logits.view([b * t, v]);
                let target_flat = target.view([b * t]);
                
                // Cross Entropy Loss
                let loss = logits_flat.cross_entropy_for_logits(&target_flat);
                
                // Backward & Step
                self.optimizer.backward_step(&loss);
                
                let loss_val = loss.double_value(&[]);
                epoch_loss += loss_val;
                
                if batch_idx % 10 == 0 {
                    println!("Epoch {} | Batch {}/{} | Loss: {:.4}", epoch, batch_idx, num_batches, loss_val);
                }
            }
            
            println!("Epoch {} Average Loss: {:.4}", epoch, epoch_loss / num_batches as f64);
            
            // Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 {
                self.save_checkpoint(epoch)?;
            }
        }
        
        Ok(())
    }

    fn save_checkpoint(&self, epoch: usize) -> Result<()> {
        let path = PathBuf::from(&self.config.checkpoint_dir);
        if !path.exists() {
            std::fs::create_dir_all(&path)?;
        }
        
        let filename = path.join(format!("checkpoint_epoch_{}.safetensors", epoch));
        // self.vs.save(&filename)?; // vs.save saves to .ot (Torch format) usually.
        // For safetensors, we might need a custom saver or just stick to torch format for now
        // to be compatible with tch. Let's use vs.save for simplistic restoration.
        self.vs.save(filename)?;
        
        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&self.model.config)?;
        std::fs::write(config_path, config_json)?;
        
        println!("Saved checkpoint and config to {:?}", path);
        Ok(())
    }
}
