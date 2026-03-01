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
    /// Maximum context window size (max sequence length the model can handle).
    pub max_seq_len: i64,
    /// Original training context length (for NTK RoPE scaling).
    /// NTK scaling factor = max_seq_len / original_max_seq_len.
    #[serde(default = "default_original_max_seq_len")]
    pub original_max_seq_len: i64,
    /// Dropout probability (applied to attention and residual connections).
    pub dropout: f64,
    /// RMSNorm epsilon value (for numerical stability).
    pub layer_norm_epsilon: f64,
    /// Whether to use bias in linear layers (typically false in modern LLMs like Llama/PaLM).
    pub use_bias: bool,

    // --- Long-Context Fields ---

    /// RoPE base frequency (default 10000.0).
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    /// Sliding window size for local attention (default 4096).
    #[serde(default = "default_window_size")]
    pub window_size: i64,
    /// Number of sink tokens pinned at position 0 (global context anchors, default 4).
    #[serde(default = "default_sink_tokens")]
    pub sink_tokens: i64,
    /// Maximum number of tokens the KV cache holds before eviction (default 32768).
    #[serde(default = "default_kv_cache_capacity")]
    pub kv_cache_capacity: usize,
}

fn default_original_max_seq_len() -> i64 { 4096 }
fn default_rope_base() -> f32 { 10000.0 }
fn default_window_size() -> i64 { 4096 }
fn default_sink_tokens() -> i64 { 4 }
fn default_kv_cache_capacity() -> usize { 32768 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            vocab_size: 50257,
            max_seq_len: 1024,
            original_max_seq_len: 4096,
            dropout: 0.0,
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            rope_base: 10000.0,
            window_size: 4096,
            sink_tokens: 4,
            kv_cache_capacity: 32768,
        }
    }
}

impl ModelConfig {
    pub fn head_size(&self) -> i64 {
        self.n_embd / self.n_head
    }

    /// Compute the NTK-scaled RoPE base frequency.
    /// This allows the model to generalize to positions beyond its training length
    /// by stretching low-frequency positional features while keeping high-frequency
    /// (local) features intact.
    ///
    /// Formula: base' = base * scale^(dim / (dim - 2))
    pub fn ntk_scaled_rope_base(&self) -> f32 {
        if self.max_seq_len <= self.original_max_seq_len {
            return self.rope_base;
        }
        let scale = self.max_seq_len as f32 / self.original_max_seq_len as f32;
        let dim = self.head_size() as f32;
        self.rope_base * scale.powf(dim / (dim - 2.0))
    }
}
