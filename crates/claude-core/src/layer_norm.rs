use tch::{nn, Tensor, Kind};
use crate::config::ModelConfig;

#[derive(Debug)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let weight = vs.var("weight", &[config.n_embd], nn::Init::Const(1.0));
        Self {
            weight,
            eps: config.layer_norm_epsilon,
        }
    }

    /// Forward pass:
    /// x: [batch, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // RMSNorm: x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        let norm = x.pow_tensor_scalar(2.0)
            .mean_dim(Some(&[-1][..]), true, Kind::Float)
            + self.eps;
        
        let output = x * norm.rsqrt();
        output * &self.weight
    }
}
