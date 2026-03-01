//! Sinusoidal positional encoding (alternative to RoPE).
//!
//! Classic "Attention Is All You Need" positional encoding.
//! Primarily kept for compatibility; RoPE is preferred for long contexts.

use tch::{Tensor, Kind, Device};

/// Generate sinusoidal positional encoding table.
///
/// Returns: [max_len, d_model] tensor.
pub fn sinusoidal_encoding(max_len: i64, d_model: i64, device: Device) -> Tensor {
    let positions = Tensor::arange(max_len, (Kind::Float, device)).unsqueeze(1);
    let dim_pairs = Tensor::arange(d_model / 2, (Kind::Float, device)).unsqueeze(0);
    let angles = &positions / Tensor::pow_scalar(10000.0, &(2.0 * &dim_pairs / d_model as f64));

    let sin_enc = angles.sin();
    let cos_enc = angles.cos();

    // Interleave sin and cos: [max_len, d_model]
    let mut encoding = Tensor::zeros(&[max_len, d_model], (Kind::Float, device));
    let _ = encoding.slice(1, 0, d_model, 2).copy_(&sin_enc);
    let _ = encoding.slice(1, 1, d_model, 2).copy_(&cos_enc);

    encoding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_shape() {
        let enc = sinusoidal_encoding(100, 64, Device::Cpu);
        assert_eq!(enc.size(), vec![100, 64]);
    }
}
