//! CUDA device utilities and memory management helpers.

use super::TensorDevice;

/// Query available CUDA devices (stub — actual implementation depends on backend).
pub fn available_devices() -> Vec<TensorDevice> {
    // In a real implementation, query the CUDA runtime
    // For now, return CPU only
    vec![TensorDevice::Cpu]
}

/// CUDA memory info for a device.
#[derive(Debug, Clone)]
pub struct CudaMemoryInfo {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub used_bytes: u64,
}

impl CudaMemoryInfo {
    /// Percentage of VRAM currently used.
    pub fn usage_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

/// Estimate VRAM required for a model.
pub fn estimate_vram_bytes(
    num_params: u64,
    bytes_per_param: u64,
    kv_cache_tokens: u64,
    num_layers: u64,
    hidden_dim: u64,
) -> u64 {
    let model_bytes = num_params * bytes_per_param;
    // KV cache: 2 (K+V) × layers × tokens × hidden_dim × bytes_per_param
    let kv_bytes = 2 * num_layers * kv_cache_tokens * hidden_dim * bytes_per_param;
    // Activation memory (rough estimate: 2x model size for gradients during training)
    model_bytes + kv_bytes
}
