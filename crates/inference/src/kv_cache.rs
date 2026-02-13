use tch::{Tensor, Device};

#[derive(Debug)]
pub struct KVCache {
    /// List of (key, value) tensors for each layer.
    /// Each tensor shape: [1, n_head, cache_len, head_size]
    /// (Usually B=1 for inference)
    pub layers: Vec<(Tensor, Tensor)>,
    pub max_seq_len: usize,
    pub device: Device,
}

impl KVCache {
    pub fn new(n_layers: usize, max_seq_len: usize, device: Device) -> Self {
        Self {
            layers: Vec::with_capacity(n_layers),
            max_seq_len,
            device,
        }
    }

    /// Add a new key/value pair to the cache for a specific layer.
    /// If the cache doesn't exist for this layer, it is initialized.
    /// k: [1, n_head, 1, head_size] (new token key)
    /// v: [1, n_head, 1, head_size] (new token value)
    pub fn update(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) {
        if layer_idx >= self.layers.len() {
            // First token, initialize cache with current k, v
            // Make sure to detach from graph!
            let k_new = k.detach().to(self.device);
            let v_new = v.detach().to(self.device);
            self.layers.push((k_new, v_new));
        } else {
            // Append along sequence dimension (dim=2)
            let (k_cache, v_cache) = &self.layers[layer_idx];
            
            // Concatenate: [1, H, L, D] + [1, H, 1, D] -> [1, H, L+1, D]
            let k_new = Tensor::cat(&[k_cache, k], 2);
            let v_new = Tensor::cat(&[v_cache, v], 2);
            
            // Check max length and truncate if needed (sliding window)
            // or return error. Typically for generation we stop or shift.
            // For now, simple update.
            
            self.layers[layer_idx] = (k_new, v_new);
        }
    }

    /// Get the key/value for a layer.
    /// Returns (k, v) or None if empty.
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx < self.layers.len() {
            Some((&self.layers[layer_idx].0, &self.layers[layer_idx].1))
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.layers.clear();
    }
}
