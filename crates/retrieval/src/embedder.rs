//! Simple embedding pipeline for RAG.
//!
//! Uses the model's embedding layer to convert text chunks into vectors.

use tch::{Tensor, Device, Kind, no_grad};

/// Trait for text embedding models.
pub trait Embedder {
    /// Embed a batch of text chunks into dense vectors.
    fn embed(&self, texts: &[&str]) -> Tensor;
    /// Embedding dimension.
    fn dim(&self) -> i64;
}

/// A simple mean-pooling embedder that uses a token embedding table directly.
pub struct MeanPoolEmbedder {
    embedding_table: Tensor,
    dim: i64,
    device: Device,
}

impl MeanPoolEmbedder {
    /// Create from a pre-trained embedding matrix [vocab_size, dim].
    pub fn new(embedding_table: Tensor, device: Device) -> Self {
        let dim = embedding_table.size()[1];
        Self {
            embedding_table: embedding_table.to(device),
            dim,
            device,
        }
    }

    /// Simple character-hash "tokenizer" for embedding.
    fn simple_encode(&self, text: &str) -> Vec<i64> {
        let vocab_size = self.embedding_table.size()[0];
        text.bytes()
            .map(|b| (b as i64) % vocab_size)
            .collect()
    }
}

impl Embedder for MeanPoolEmbedder {
    fn embed(&self, texts: &[&str]) -> Tensor {
        no_grad(|| {
            let mut embeddings = Vec::with_capacity(texts.len());

            for text in texts {
                let token_ids = self.simple_encode(text);
                if token_ids.is_empty() {
                    embeddings.push(Tensor::zeros(&[self.dim], (Kind::Float, self.device)));
                    continue;
                }

                let ids = Tensor::from_slice(&token_ids).to(self.device);
                let token_embeds = self.embedding_table.index_select(0, &ids);
                let mean_embed = token_embeds.mean_dim(Some(&[0i64][..]), false, Kind::Float);
                embeddings.push(mean_embed);
            }

            Tensor::stack(&embeddings, 0)
        })
    }

    fn dim(&self) -> i64 {
        self.dim
    }
}
