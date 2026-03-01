//! INT8 quantization for model weights.
//!
//! Symmetric per-tensor quantization: FP32 → INT8
//! Each weight tensor gets a single scale factor.

use super::QuantStats;

/// A quantized INT8 tensor (stored as Vec<i8> + scale).
#[derive(Debug, Clone)]
pub struct QuantizedTensorI8 {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
    pub stats: QuantStats,
}

impl QuantizedTensorI8 {
    /// Quantize a slice of f32 values to INT8 (symmetric).
    pub fn quantize(values: &[f32], shape: Vec<usize>) -> Self {
        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let stats = QuantStats::symmetric(min_val, max_val, 8);

        let data: Vec<i8> = values
            .iter()
            .map(|&v| (v / stats.scale).round().clamp(-128.0, 127.0) as i8)
            .collect();

        Self { data, shape, stats }
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| q as f32 * self.stats.scale)
            .collect()
    }

    /// Memory savings ratio vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        4.0 // f32 (4 bytes) → i8 (1 byte) = 4x compression
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<QuantStats>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_q8() {
        let values: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let quantized = QuantizedTensorI8::quantize(&values, vec![5]);
        let dequantized = quantized.dequantize();

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.02, "orig={}, deq={}", orig, deq);
        }
    }
}
