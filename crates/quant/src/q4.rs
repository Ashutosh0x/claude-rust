//! 4-bit quantization for aggressive model compression.
//!
//! Packs two 4-bit values into each byte for 8x compression over FP32.
//! Uses symmetric quantization with per-group scaling (group size = 128).


/// Group size for 4-bit quantization (number of elements sharing a scale).
pub const Q4_GROUP_SIZE: usize = 128;

/// A quantized Q4 tensor. Two 4-bit values packed per byte.
#[derive(Debug, Clone)]
pub struct QuantizedTensorQ4 {
    /// Packed data: each byte holds two 4-bit values (low nibble first).
    pub data: Vec<u8>,
    /// One scale per group of Q4_GROUP_SIZE elements.
    pub scales: Vec<f32>,
    pub shape: Vec<usize>,
    pub num_elements: usize,
}

impl QuantizedTensorQ4 {
    /// Quantize f32 values to 4-bit with per-group scaling.
    pub fn quantize(values: &[f32], shape: Vec<usize>) -> Self {
        let num_elements = values.len();
        let num_groups = (num_elements + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut packed = Vec::with_capacity((num_elements + 1) / 2);

        for group_start in (0..num_elements).step_by(Q4_GROUP_SIZE) {
            let group_end = (group_start + Q4_GROUP_SIZE).min(num_elements);
            let group = &values[group_start..group_end];

            // Symmetric: scale = max(|v|) / 7
            let abs_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
            scales.push(scale);

            // Quantize to [-8, 7] range, then offset to [0, 15]
            let quantized: Vec<u8> = group
                .iter()
                .map(|&v| {
                    let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
                    (q + 8) as u8 // offset to unsigned [0, 15]
                })
                .collect();

            // Pack two nibbles per byte
            for pair in quantized.chunks(2) {
                let lo = pair[0] & 0x0F;
                let hi = if pair.len() > 1 { pair[1] & 0x0F } else { 0 };
                packed.push(lo | (hi << 4));
            }
        }

        Self {
            data: packed,
            scales,
            shape,
            num_elements,
        }
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.num_elements);
        let mut elem_idx = 0;
        let mut byte_idx = 0;

        for (group_idx, &scale) in self.scales.iter().enumerate() {
            let group_end = ((group_idx + 1) * Q4_GROUP_SIZE).min(self.num_elements);

            while elem_idx < group_end {
                let byte = self.data[byte_idx];
                let lo = (byte & 0x0F) as i8 - 8; // un-offset
                result.push(lo as f32 * scale);
                elem_idx += 1;

                if elem_idx < group_end {
                    let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                    result.push(hi as f32 * scale);
                    elem_idx += 1;
                }
                byte_idx += 1;
            }
        }

        result
    }

    /// Memory savings ratio vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        let fp32_bytes = self.num_elements * 4;
        let q4_bytes = self.data.len() + self.scales.len() * 4;
        fp32_bytes as f64 / q4_bytes as f64
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_q4() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let quantized = QuantizedTensorQ4::quantize(&values, vec![256]);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.len(), values.len());
        // Q4 is lossy — expect ~0.1 max error for [-1, 1] range
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.15, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let values: Vec<f32> = vec![0.0; 1024];
        let quantized = QuantizedTensorQ4::quantize(&values, vec![1024]);
        assert!(quantized.compression_ratio() > 6.0); // ~7-8x expected
    }
}
