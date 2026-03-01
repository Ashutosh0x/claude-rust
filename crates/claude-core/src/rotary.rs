use tch::{Tensor, Kind, Device};
use crate::config::ModelConfig;

/// NTK-Aware Rotary Position Embedding.
///
/// Precomputes a frequency table at construction using the NTK-scaled base,
/// then applies rotary embeddings by indexing into the table with explicit
/// position indices. This is essential for correctness with evicting KV caches
/// where token positions are non-contiguous.
pub struct RotaryEmbedding {
    /// Precomputed cos values: [max_seq_len, head_dim]
    cos_cache: Tensor,
    /// Precomputed sin values: [max_seq_len, head_dim]
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding with NTK-aware scaling.
    ///
    /// The frequency table is precomputed for positions [0, max_seq_len).
    /// NTK scaling adjusts the base frequency so that high-frequency (local)
    /// features stay intact while low-frequency (positional) features stretch
    /// to cover the extended context length.
    pub fn new(config: &ModelConfig, device: Device) -> Self {
        let head_dim = config.head_size();
        let max_seq_len = config.max_seq_len;
        let scaled_base = config.ntk_scaled_rope_base();

        // inv_freq_i = 1.0 / (scaled_base ^ (2i / head_dim))
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / scaled_base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq).to(device);

        // positions: [max_seq_len]
        let positions = Tensor::arange(max_seq_len, (Kind::Float, device));

        // freqs: [max_seq_len, half_dim] = outer product of positions and inv_freq
        let freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0);

        // cos_cache, sin_cache: [max_seq_len, head_dim] (duplicate for both halves)
        let cos_cache = Tensor::cat(&[&freqs.cos(), &freqs.cos()], -1);
        let sin_cache = Tensor::cat(&[&freqs.sin(), &freqs.sin()], -1);

        Self {
            cos_cache,
            sin_cache,
        }
    }

    /// Apply rotary embeddings to a tensor using explicit position indices.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, n_head, seq_len, head_dim]
    /// * `positions` - Position indices of shape [seq_len] (absolute positions in the sequence)
    ///
    /// # Returns
    /// Tensor of same shape with rotary embeddings applied.
    pub fn apply(&self, x: &Tensor, positions: &Tensor) -> Tensor {
        // Gather cos/sin for the requested positions: [seq_len, head_dim]
        let cos = self.cos_cache.index_select(0, positions); // [seq_len, head_dim]
        let sin = self.sin_cache.index_select(0, positions); // [seq_len, head_dim]

        // Reshape to [1, 1, seq_len, head_dim] for broadcasting
        let cos = cos.unsqueeze(0).unsqueeze(0);
        let sin = sin.unsqueeze(0).unsqueeze(0);

        // Rotary transform: (x * cos) + (rotate_half(x) * sin)
        (x * &cos) + (&Self::rotate_half(x) * &sin)
    }

    /// Convenience: apply with a contiguous range of positions starting from `offset`.
    ///
    /// Equivalent to `apply(x, [offset, offset+1, ..., offset+seq_len-1])`.
    pub fn apply_with_offset(&self, x: &Tensor, offset: i64) -> Tensor {
        let seq_len = x.size()[2];
        let device = x.device();
        let positions = Tensor::arange_start(offset, offset + seq_len, (Kind::Int64, device));
        self.apply(x, &positions)
    }

    /// Rotate the second half of the last dimension: [-x2, x1]
    fn rotate_half(x: &Tensor) -> Tensor {
        let x_size = x.size();
        let last_dim = x_size[x_size.len() - 1];
        let half = last_dim / 2;

        let x1 = x.narrow(-1, 0, half);
        let x2 = x.narrow(-1, half, half);

        Tensor::cat(&[&(-&x2), &x1], -1)
    }
}
