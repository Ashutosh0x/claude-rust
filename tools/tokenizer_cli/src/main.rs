use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizer::{Trainer, BPE};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new BPE tokenizer
    Train {
        /// Input files to train on
        #[arg(required = true)]
        files: Vec<String>,

        /// Output directory for vocab.json and merges.txt
        #[arg(short, long, default_value = "data/processed")]
        output_dir: PathBuf,

        /// Vocabulary size
        #[arg(short, long, default_value_t = 32000)]
        vocab_size: usize,

        /// Minimum frequency for a pair to be merged
        #[arg(long, default_value_t = 2)]
        min_frequency: u32,
    },
    /// Encode text using existing tokenizer
    Encode {
        /// Path to vocab.json
        #[arg(long)]
        vocab: PathBuf,
        
        /// Path to merges.txt
        #[arg(long)]
        merges: PathBuf,

        /// Text to encode
        #[arg(short, long)]
        text: String,
    },
    /// Decode IDs using existing tokenizer
    Decode {
        /// Path to vocab.json
        #[arg(long)]
        vocab: PathBuf,
        
        /// Path to merges.txt
        #[arg(long)]
        merges: PathBuf,

        /// IDs to decode (comma separated)
        #[arg(short, long)]
        ids: String,
    },
}

fn save_merges(merges: &HashMap<(String, String), u32>, path: impl AsRef<Path>) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "#version: 0.2")?;
    // We need to sort merges by rank to save logically, though HashMap iteration order is random.
    // In BPE loading, we use line number as rank. So saving must be sorted by rank.
    let mut sorted_merges: Vec<_> = merges.iter().collect();
    sorted_merges.sort_by_key(|&(_, rank)| rank);

    for ((p1, p2), _) in sorted_merges {
        writeln!(file, "{} {}", p1, p2)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            files,
            output_dir,
            vocab_size,
            min_frequency,
        } => {
            println!("Training tokenizer on {:?}...", files);
            let trainer = Trainer::new(vocab_size, min_frequency, vec!["<UNK>".to_string(), "<PAD>".to_string(), "<EOS>".to_string()]);
            // Convert String paths to &str
            // trainer.train expects &[String]
            match trainer.train(&files) {
                Ok(bpe) => {
                    fs::create_dir_all(&output_dir)?;
                    let vocab_path = output_dir.join("vocab.json");
                    let merges_path = output_dir.join("merges.txt");

                    println!("Saving vocab to {:?}", vocab_path);
                    bpe.vocab.save(&vocab_path).context("Failed to save vocab")?;

                    println!("Saving merges to {:?}", merges_path);
                    save_merges(&bpe.merges, &merges_path).context("Failed to save merges")?;
                    
                    println!("Training complete.");
                }
                Err(e) => {
                    eprintln!("Error during training: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Encode { vocab, merges, text } => {
            let bpe = BPE::from_files(vocab, merges).context("Failed to load tokenizer")?;
            let ids = bpe.encode(&text);
            println!("Encoded IDs: {:?}", ids);
        }
        Commands::Decode { vocab, merges, ids } => {
            let bpe = BPE::from_files(vocab, merges).context("Failed to load tokenizer")?;
            let id_list: Vec<u32> = ids
                .split(',')
                .map(|s| s.trim().parse().expect("Invalid ID"))
                .collect();
            let text = bpe.decode(&id_list);
            println!("Decoded text: {}", text);
        }
    }

    Ok(())
}
