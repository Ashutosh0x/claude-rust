use tch::{Tensor, Kind, Device};

pub struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: i64, device: Device) -> Self {
        // inv_freq = 1.0 / (10000 ^ (2i / dim))
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (10000.0f32.powf(i as f32 / dim as f32)))
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq).to(device);
        
        Self { inv_freq }
    }

    /// x: [batch, n_head, seq_len, head_dim]
    pub fn forward(&self, x: &Tensor, seq_len: i64) -> Tensor {
        let device = x.device();
        let t = Tensor::arange(seq_len, (Kind::Float, device));
        
        // freqs: [seq_len, dim/2]
        let freqs = t.outer(&self.inv_freq);
        
        // emb: [seq_len, dim] -> [1, 1, seq_len, dim]
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        let emb = emb.unsqueeze(0).unsqueeze(0);
        
        // cos, sin
        let cos = emb.cos();
        let sin = emb.sin();
        
        // rotary transform: (x * cos) + (rotate_half(x) * sin)
        (x * &cos) + (&Self::rotate_half(x) * &sin)
    }

    fn rotate_half(x: &Tensor) -> Tensor {
        let x_size = x.size();
        let last_dim = x_size[x_size.len() - 1];
        let half = last_dim / 2;
        
        let x1 = x.narrow(-1, 0, half);
        let x2 = x.narrow(-1, half, half);
        
        Tensor::cat(&[&-x2, &x1], -1)
    }
}
