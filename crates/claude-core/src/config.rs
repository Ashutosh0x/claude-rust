use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Dimension of the token embeddings (and internal transformer states).
    pub n_embd: i64,
    /// Number of attention heads.
    pub n_head: i64,
    /// Number of transformer layers.
    pub n_layer: i64,
    /// Size of the vocabulary.
    pub vocab_size: i64,
    /// Maximum context window size (max sequence length).
    pub max_seq_len: i64,
    /// Dropout probability (applied to attention and residual connections).
    pub dropout: f64,
    /// RMSNorm epsilon value (for numerical stability).
    pub layer_norm_epsilon: f64,
    /// Whether to use bias in linear layers (typically false in modern LLMs like Llama/PaLM).
    pub use_bias: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_embd: 768, // GPT-2 Small equivalent
            n_head: 12,
            n_layer: 12,
            vocab_size: 50257,
            max_seq_len: 2048,
            dropout: 0.0,
            layer_norm_epsilon: 1e-5,
            use_bias: false,
        }
    }
}

impl ModelConfig {
    pub fn head_size(&self) -> i64 {
        self.n_embd / self.n_head
    }
}
