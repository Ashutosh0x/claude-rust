use anyhow::Result;
use std::fs;
use std::path::Path;
use tch::Device;

use claude_core::ModelConfig;
use tokenizer::{BPE, Trainer as TokenizerTrainer};
use trainer::{Trainer, TrainerConfig};

fn main() -> Result<()> {
    env_logger::init();
    
    let dataset_path = "data/claude_system_prompts.txt";
    let vocab_path = "data/vocab.json";
    
    // 1. Train or Load Tokenizer
    let tokenizer = if Path::new(vocab_path).exists() {
        println!("Loading existing tokenizer from {}", vocab_path);
        BPE::load(vocab_path)?
    } else {
        println!("Training new tokenizer on {}", dataset_path);
        let trainer = TokenizerTrainer::new(500, 1, vec!["<pad>".to_string(), "<unk>".to_string(), "<s>".to_string(), "</s>".to_string()]);
        let bpe = trainer.train(&[dataset_path.to_string()])?;
        bpe.save(vocab_path)?;
        bpe
    };

    // 2. Load Configs from configs/
    let model_config_path = "configs/model_config.yaml";
    let training_config_path = "configs/training_config.yaml";
    
    let mut model_config: ModelConfig = if Path::new(model_config_path).exists() {
        let content = fs::read_to_string(model_config_path)?;
        serde_yaml::from_str(&content)?
    } else {
        ModelConfig::default()
    };
    // Ensure vocab size matches the recently trained/loaded tokenizer
    model_config.vocab_size = tokenizer.vocab.len() as i64;

    let trainer_config: TrainerConfig = if Path::new(training_config_path).exists() {
        let content = fs::read_to_string(training_config_path)?;
        serde_yaml::from_str(&content)?
    } else {
        TrainerConfig::default()
    };
    
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let mut trainer = Trainer::new(model_config, trainer_config, device)?;
    
    // 4. Load Data
    let text = fs::read_to_string(dataset_path)?;
    
    // 5. Train
    trainer.train(&text, &tokenizer)?;
    
    println!("Training complete!");
    
    Ok(())
}
