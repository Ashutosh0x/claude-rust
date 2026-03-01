//! Token and position embedding layer.

use tch::{nn, Tensor};
use crate::config::ModelConfig;

/// Combined token + position embedding.
pub struct Embedding {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
}

impl Embedding {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let token_embedding = nn::embedding(
            vs / "token_embedding",
            config.vocab_size,
            config.n_embd,
            Default::default(),
        );
        let position_embedding = nn::embedding(
            vs / "position_embedding",
            config.max_seq_len,
            config.n_embd,
            Default::default(),
        );
        Self {
            token_embedding,
            position_embedding,
        }
    }

    /// Forward: token_ids [B, T] → embeddings [B, T, C]
    pub fn forward(&self, token_ids: &Tensor) -> Tensor {
        let seq_len = token_ids.size()[1];
        let device = token_ids.device();
        let positions = Tensor::arange(seq_len, (tch::Kind::Int64, device)).unsqueeze(0);
        let tok_emb = self.token_embedding.forward(token_ids);
        let pos_emb = self.position_embedding.forward(&positions);
        tok_emb + pos_emb
    }
}
