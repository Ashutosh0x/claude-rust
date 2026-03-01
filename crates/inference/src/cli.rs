//! CLI interface for inference (standalone binary mode).

use clap::Parser;

/// Claude-Rust Inference CLI
#[derive(Parser, Debug)]
#[command(name = "claude-infer", about = "Run inference on Claude-Rust models")]
pub struct InferenceCli {
    /// Path to model checkpoint directory.
    #[arg(short, long, default_value = "checkpoints")]
    pub checkpoint_dir: String,

    /// Path to tokenizer vocabulary file.
    #[arg(short, long, default_value = "data/vocab.json")]
    pub vocab_path: String,

    /// Prompt text to generate from.
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Maximum tokens to generate.
    #[arg(short, long, default_value = "100")]
    pub max_tokens: usize,

    /// Sampling temperature.
    #[arg(short, long, default_value = "0.8")]
    pub temperature: f64,

    /// Top-p (nucleus) sampling.
    #[arg(long, default_value = "0.9")]
    pub top_p: f64,

    /// Start HTTP server instead of CLI generation.
    #[arg(long)]
    pub serve: bool,

    /// Server port (when --serve is used).
    #[arg(long, default_value = "8000")]
    pub port: u16,
}
