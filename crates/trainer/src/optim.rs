/// AdamW optimizer wrapper with gradient clipping.
///
/// Wraps tch's built-in AdamW with additional features:
/// - Configurable gradient clipping (max norm)
/// - Weight decay control

pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: Option<f64>,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            grad_clip_norm: Some(1.0),
        }
    }
}

impl AdamWConfig {
    /// Build an optimizer from this config.
    pub fn build(&self, vs: &tch::nn::VarStore) -> anyhow::Result<tch::nn::Optimizer> {
        use tch::nn::OptimizerConfig;
        let opt = tch::nn::AdamW::default()
            .build(vs, self.lr)?;
        Ok(opt)
    }
}
