//! Tensor backend abstraction layer.
//!
//! Provides a unified interface over different tensor backends
//! (tch/LibTorch, burn, ndarray) to avoid lock-in.

pub mod backend;
pub mod tensor_ops;
pub mod cuda;

/// Supported tensor backends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    /// LibTorch via tch-rs (default, GPU-accelerated).
    Tch,
    /// Burn framework (wgpu/Vulkan/Metal).
    Burn,
    /// ndarray (CPU-only, for prototyping).
    NdArray,
}

/// Device abstraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDevice {
    Cpu,
    Cuda(usize),
}

impl TensorDevice {
    /// Check if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(self, TensorDevice::Cuda(_))
    }
}

impl Default for TensorDevice {
    fn default() -> Self {
        TensorDevice::Cpu
    }
}
