use tch::{Tensor, Kind, IndexOp};
use rand::distributions::Distribution;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
        }
    }
}

pub struct Sampler;

impl Sampler {
    /// Sample a token ID from logits.
    /// logits: [vocab_size] tensor.
    /// history: slice of previously generated token IDs.
    pub fn sample(logits: &Tensor, params: &SamplingParams, history: &[i64]) -> anyhow::Result<i64> {
        let _guard = tch::no_grad_guard();

        // 0. Repetition Penalty
        let logits = if params.repetition_penalty != 1.0 && !history.is_empty() {
            use std::collections::HashSet;
            let unique_tokens: HashSet<_> = history.iter().collect();
            let l = logits.to_device(tch::Device::Cpu);
            for &&token_id in &unique_tokens {
                if token_id < 0 { continue; } // Safety
                let current_val = l.double_value(&[token_id]);
                let new_val = if current_val < 0.0 {
                    current_val * params.repetition_penalty
                } else {
                    current_val / params.repetition_penalty
                };
                let _ = l.i(token_id).fill_(new_val);
            }
            l
        } else {
            logits.shallow_clone()
        };

        // 1. Temperature scaling
        if params.temperature < 1e-5 {
            return Ok(logits.argmax(0, false).int64_value(&[]));
        }

        let scaled_logits = logits / params.temperature;
        
        // 2. Softmax for probabilities
        let probs = scaled_logits.softmax(-1, Kind::Float);

        // 3. Top-K filtering
        // We create a mask where indices NOT in top-k are zeroed out.
        // Actually, let's just use the distribution logic directly on vectors for CPU-based multinomial.
        // Tch doesn't expose easy WeightedIndex on GPU directly in safe Rust without boilerplate.
        // CPU fallback is fine for inference (vocab size < 100k).
        
        let probs_vec: Vec<f64> = Vec::<f64>::try_from(&probs)?;
        
        // Convert to (prob, index) tuples
        let mut candidates: Vec<(f64, usize)> = probs_vec
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
            
        // Sort descending by probability
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Top-K Cutoff
        if params.top_k > 0 && params.top_k < candidates.len() {
            candidates.truncate(params.top_k);
        }

        // 5. Top-P (Nucleus) Cutoff
        if params.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cutoff_index = candidates.len() - 1;
            
            for (i, (p, _)) in candidates.iter().enumerate() {
                cumulative += p;
                if cumulative > params.top_p {
                    cutoff_index = i;
                    break;
                }
            }
            candidates.truncate(cutoff_index + 1);
        }
        
        // 6. Renormalize remaining probabilities
        let sum_p: f64 = candidates.iter().map(|(p, _)| p).sum();
        let renorm_probs: Vec<f64> = candidates.iter().map(|(p, _)| p / sum_p).collect();
        
        // 7. Sample
        let dist = rand::distributions::WeightedIndex::new(&renorm_probs)
            .map_err(|e| anyhow::anyhow!("WeightedIndex error: {}", e))?;
            
        let mut rng = rand::thread_rng();
        let sampled_idx_in_subset = dist.sample(&mut rng);
        let global_idx = candidates[sampled_idx_in_subset].1;

        Ok(global_idx as i64)
    }
}
