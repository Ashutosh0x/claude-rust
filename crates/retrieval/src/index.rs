//! Flat index for nearest-neighbor search.
//!
//! A simple brute-force index using cosine similarity.
//! For large-scale deployments, swap with FAISS or HNSW.

use tch::{Tensor, Device};

/// Flat (brute-force) index for similarity search.
pub struct FlatIndex {
    vectors: Option<Tensor>,
    dim: i64,
    device: Device,
}

impl FlatIndex {
    pub fn new(dim: i64, device: Device) -> Self {
        Self {
            vectors: None,
            dim,
            device,
        }
    }

    /// Add vectors to the index. Each row is one vector.
    pub fn add(&mut self, vectors: &Tensor) {
        let v = vectors.to(self.device);
        match &mut self.vectors {
            Some(existing) => {
                *existing = Tensor::cat(&[existing.shallow_clone(), v], 0);
            }
            None => {
                self.vectors = Some(v);
            }
        }
    }

    /// Search for the top-k most similar vectors.
    /// Returns (scores, indices) tensors.
    pub fn search(&self, query: &Tensor, top_k: usize) -> Option<(Tensor, Tensor)> {
        let vectors = self.vectors.as_ref()?;
        let n = vectors.size()[0] as usize;
        let k = top_k.min(n);

        let q = query.to(self.device).view([1, self.dim]);

        // L2 normalize for cosine similarity
        let q_norm = &q / (q.norm_scalaropt_dim(2, &[1], true) + 1e-8);
        let v_norm = vectors / (vectors.norm_scalaropt_dim(2, &[1], true) + 1e-8);

        let scores = q_norm.matmul(&v_norm.transpose(0, 1)).view([-1]);
        let (top_scores, top_indices) = scores.topk(k as i64, 0, true, true);

        Some((top_scores, top_indices))
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors
            .as_ref()
            .map(|v| v.size()[0] as usize)
            .unwrap_or(0)
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
