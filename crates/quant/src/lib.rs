//! Quantization utilities for reducing model memory footprint.
//!
//! Provides INT8 and Q4 quantization for inference on consumer hardware.

pub mod q8;
pub mod q4;

/// Quantization format metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantFormat {
    /// Full precision (no quantization)
    FP32,
    /// 8-bit integer quantization
    INT8,
    /// 4-bit quantization (GPTQ-style)
    Q4,
}

/// Statistics used for quantization calibration.
#[derive(Debug, Clone)]
pub struct QuantStats {
    pub min_val: f32,
    pub max_val: f32,
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantStats {
    /// Compute symmetric quantization parameters.
    pub fn symmetric(min_val: f32, max_val: f32, num_bits: u32) -> Self {
        let abs_max = min_val.abs().max(max_val.abs());
        let qmax = (1 << (num_bits - 1)) as f32 - 1.0;
        let scale = abs_max / qmax;
        Self {
            min_val,
            max_val,
            scale: if scale == 0.0 { 1.0 } else { scale },
            zero_point: 0,
        }
    }

    /// Compute asymmetric quantization parameters.
    pub fn asymmetric(min_val: f32, max_val: f32, num_bits: u32) -> Self {
        let qmax = ((1u64 << num_bits) - 1) as f32;
        let scale = (max_val - min_val) / qmax;
        let zero_point = (-min_val / scale).round() as i32;
        Self {
            min_val,
            max_val,
            scale: if scale == 0.0 { 1.0 } else { scale },
            zero_point,
        }
    }
}
