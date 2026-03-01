//! Language model head — projects hidden states to vocabulary logits.

use tch::{nn, Tensor};
use crate::config::ModelConfig;

/// Linear projection from hidden_dim → vocab_size (no bias, weight-tied).
pub struct LMHead {
    proj: nn::Linear,
}

impl LMHead {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let proj = nn::linear(
            vs / "lm_head",
            config.n_embd,
            config.vocab_size,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        Self { proj }
    }

    /// Forward: hidden [B, T, C] → logits [B, T, V]
    pub fn forward(&self, hidden: &Tensor) -> Tensor {
        hidden.apply(&self.proj)
    }
}
