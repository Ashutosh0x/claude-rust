use tch::{Tensor, Kind, Device};
use tokenizer::BPE;
use rand::{thread_rng, Rng};

pub struct TextDataset {
    tokens: Vec<i64>,
    context_length: usize,
    device: Device,
}

impl TextDataset {
    pub fn new(text: &str, tokenizer: &BPE, context_length: usize, device: Device) -> Self {
        let tokens: Vec<i64> = tokenizer.encode(text)
            .into_iter()
            .map(|t| t as i64)
            .collect();
        
        Self {
            tokens,
            context_length,
            device,
        }
    }

    /// Returns a batch of size `batch_size`.
    /// Each item is (input, target) where:
    /// input: [batch_size, context_length]
    /// target: [batch_size, context_length] (shifted by 1)
    pub fn sample_batch(&self, batch_size: usize) -> (Tensor, Tensor) {
        let max_start = self.tokens.len().saturating_sub(self.context_length + 1);
        if max_start == 0 {
            // Not enough data, return empty or handle gracefully
            // For now, just panic or return zero tensors if really small
            if self.tokens.len() <= 1 {
                return (
                    Tensor::zeros(&[batch_size as i64, self.context_length as i64], (Kind::Int64, self.device)),
                    Tensor::zeros(&[batch_size as i64, self.context_length as i64], (Kind::Int64, self.device))
                );
            }
        }

        let mut inputs = Vec::with_capacity(batch_size * self.context_length);
        let mut targets = Vec::with_capacity(batch_size * self.context_length);

        let mut rng = thread_rng();

        for _ in 0..batch_size {
            let start_idx = rng.gen_range(0..max_start);
            let end_idx = start_idx + self.context_length;
            
            let chunk = &self.tokens[start_idx..end_idx + 1];
            
            inputs.extend_from_slice(&chunk[0..self.context_length]);
            targets.extend_from_slice(&chunk[1..self.context_length + 1]);
        }

        let input_tensor = Tensor::from_slice(&inputs)
            .view([batch_size as i64, self.context_length as i64])
            .to(self.device);
            
        let target_tensor = Tensor::from_slice(&targets)
            .view([batch_size as i64, self.context_length as i64])
            .to(self.device);

        (input_tensor, target_tensor)
    }
}
