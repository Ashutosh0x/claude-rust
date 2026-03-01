/// Learning rate schedulers for training.
///
/// Supports linear warmup followed by cosine decay, which is the standard
/// schedule for transformer pre-training.

/// Cosine decay scheduler with optional linear warmup.
pub struct CosineScheduler {
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl CosineScheduler {
    pub fn new(base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }

    /// Get the learning rate for a given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup: 0 → base_lr
            self.base_lr * (step as f64 / self.warmup_steps.max(1) as f64)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay: base_lr → min_lr
            let progress = (step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps).max(1) as f64;
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        }
    }
}

/// Constant learning rate (no scheduling).
pub struct ConstantScheduler {
    lr: f64,
}

impl ConstantScheduler {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }

    pub fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

/// Apply the scheduled learning rate to an optimizer.
pub fn apply_lr(optimizer: &mut tch::nn::Optimizer, lr: f64) {
    optimizer.set_lr(lr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_warmup() {
        let sched = CosineScheduler::new(1e-3, 1e-5, 100, 1000);
        // At step 0: should be 0
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-10);
        // At step 50 (mid warmup): should be ~0.5e-3
        assert!((sched.get_lr(50) - 5e-4).abs() < 1e-10);
        // At step 100 (end warmup): should be base_lr
        assert!((sched.get_lr(100) - 1e-3).abs() < 1e-10);
        // At step 1000 (end): should be min_lr
        assert!((sched.get_lr(1000) - 1e-5).abs() < 1e-8);
    }
}
