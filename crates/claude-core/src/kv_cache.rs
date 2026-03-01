use tch::{Tensor, Device, Kind};

/// Evicting KV Cache with Ring Buffer and Sink Token Pinning.
///
/// For long-context inference (100K+ tokens), a naive KV cache that grows
/// linearly would exhaust GPU memory. This cache has a fixed capacity
/// and uses an eviction policy:
///
/// - **Sink tokens** (first `sink_size` positions) are always pinned —
///   they act as global context anchors that stabilize attention.
/// - **Rolling window** — the most recent tokens are kept.
/// - **Middle eviction** — when full, tokens between sinks and the recent
///   window are evicted first.
///
/// Layout: `[sink_tokens | rolling_recent_window]`
pub struct EvictingKVCache {
    /// Max tokens the cache can hold (e.g., 32768).
    pub max_capacity: usize,
    /// Number of pinned sink tokens at position 0 (e.g., 4).
    pub sink_size: usize,
    /// Per-layer key caches: [batch, n_heads, max_capacity, head_dim]
    pub key_cache: Vec<Tensor>,
    /// Per-layer value caches: [batch, n_heads, max_capacity, head_dim]
    pub value_cache: Vec<Tensor>,
    /// Current number of valid tokens in the cache.
    pub current_len: usize,
    /// Total tokens processed so far (monotonically increasing, used for RoPE positions).
    pub total_tokens_seen: usize,
    /// Number of layers.
    n_layers: usize,
}

impl EvictingKVCache {
    pub fn new(
        n_layers: usize,
        batch: i64,
        n_heads: i64,
        head_dim: i64,
        max_capacity: usize,
        sink_size: usize,
        device: Device,
    ) -> Self {
        assert!(
            sink_size < max_capacity,
            "sink_size ({}) must be less than max_capacity ({})",
            sink_size, max_capacity
        );

        let key_cache: Vec<Tensor> = (0..n_layers)
            .map(|_| {
                Tensor::zeros(
                    &[batch, n_heads, max_capacity as i64, head_dim],
                    (Kind::Float, device),
                )
            })
            .collect();

        let value_cache: Vec<Tensor> = (0..n_layers)
            .map(|_| {
                Tensor::zeros(
                    &[batch, n_heads, max_capacity as i64, head_dim],
                    (Kind::Float, device),
                )
            })
            .collect();

        Self {
            max_capacity,
            sink_size,
            key_cache,
            value_cache,
            current_len: 0,
            total_tokens_seen: 0,
            n_layers,
        }
    }

    /// Append new KV pairs for a specific layer.
    ///
    /// If the cache has space, new tokens are written directly.
    /// If full, middle tokens are evicted while sink tokens and the most
    /// recent window are preserved.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `new_k` - New key tensor: [batch, n_heads, new_tokens, head_dim]
    /// * `new_v` - New value tensor: same shape as new_k
    pub fn append(&mut self, layer: usize, new_k: &Tensor, new_v: &Tensor) {
        let new_tokens = new_k.size()[2] as usize;

        if self.current_len + new_tokens <= self.max_capacity {
            // Cache has space — write directly
            let start = self.current_len as i64;
            let _ = self.key_cache[layer]
                .narrow(2, start, new_tokens as i64)
                .copy_(new_k);
            let _ = self.value_cache[layer]
                .narrow(2, start, new_tokens as i64)
                .copy_(new_v);

            // Only update bookkeeping on the first layer to avoid double-counting
            if layer == 0 {
                self.current_len += new_tokens;
                self.total_tokens_seen += new_tokens;
            }
        } else {
            // EVICTION: shift the rolling window, keep sinks + recent + new
            let available_for_recent = self.max_capacity - self.sink_size - new_tokens;

            // Extract sink tokens (always at the start)
            let sink_k = self.key_cache[layer].narrow(2, 0, self.sink_size as i64);
            let sink_v = self.value_cache[layer].narrow(2, 0, self.sink_size as i64);

            // Extract the most recent `available_for_recent` tokens from the current cache
            let recent_start = (self.current_len - available_for_recent) as i64;
            let recent_k = self.key_cache[layer].narrow(2, recent_start, available_for_recent as i64);
            let recent_v = self.value_cache[layer].narrow(2, recent_start, available_for_recent as i64);

            // Concatenate: [sink | recent | new]
            let new_key_row = Tensor::cat(&[&sink_k, &recent_k, new_k], 2);
            let new_val_row = Tensor::cat(&[&sink_v, &recent_v, new_v], 2);

            // Write back into the pre-allocated buffer
            let _ = self.key_cache[layer]
                .narrow(2, 0, self.max_capacity as i64)
                .copy_(&new_key_row);
            let _ = self.value_cache[layer]
                .narrow(2, 0, self.max_capacity as i64)
                .copy_(&new_val_row);

            if layer == 0 {
                self.current_len = self.max_capacity;
                self.total_tokens_seen += new_tokens;
            }
        }
    }

    /// Get the valid portion of the KV cache for a layer.
    ///
    /// Returns (key, value) tensors sliced to `[batch, n_heads, current_len, head_dim]`.
    pub fn get_view(&self, layer: usize) -> (Tensor, Tensor) {
        let k = self.key_cache[layer].narrow(2, 0, self.current_len as i64);
        let v = self.value_cache[layer].narrow(2, 0, self.current_len as i64);
        (k, v)
    }

    /// Reset the cache (for a new sequence).
    pub fn clear(&mut self) {
        self.current_len = 0;
        self.total_tokens_seen = 0;
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}
