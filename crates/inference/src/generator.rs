use tch::{Tensor, Device, IndexOp};
use claude_core::ClaudeTransformer;
use claude_core::kv_cache::EvictingKVCache;
use crate::sampling::{Sampler, SamplingParams};

use std::sync::Arc;

pub struct Generator {
    model: Arc<ClaudeTransformer>,
    device: Device,
}

impl Generator {
    pub fn new(model: Arc<ClaudeTransformer>, device: Device) -> Self {
        Self { model, device }
    }

    /// Create an EvictingKVCache sized to this model's config.
    fn create_cache(&self, batch_size: i64) -> EvictingKVCache {
        let config = &self.model.config;
        EvictingKVCache::new(
            config.n_layer as usize,
            batch_size,
            config.n_head,
            config.head_size(),
            config.kv_cache_capacity,
            config.sink_tokens as usize,
            self.device,
        )
    }

    pub fn generate_stream(
        &mut self,
        prompt_ids: &[i64],
        max_new_tokens: usize,
        params: &SamplingParams,
        tx: tokio::sync::mpsc::Sender<i64>,
    ) -> anyhow::Result<()> {
        let mut tokens = prompt_ids.to_vec();

        // Initialize evicting KV cache
        let mut cache = self.create_cache(1);

        // 1. Prefill — process entire prompt in one pass
        let input_tensor = Tensor::from_slice(&tokens)
            .view([1, tokens.len() as i64])
            .to(self.device);
        let logits = self.model.forward(&input_tensor, Some(&mut cache));

        // Sample first new token from last position
        let next_token_logits = logits.i((0, -1, ..));
        let mut next_token = Sampler::sample(&next_token_logits, params, &tokens)?;

        let _ = tx.blocking_send(next_token);
        tokens.push(next_token);

        // 2. Decode loop — one token at a time, cache handles eviction
        for _ in 0..max_new_tokens {
            let input_tensor = Tensor::from_slice(&[next_token])
                .view([1, 1])
                .to(self.device);
            let logits = self.model.forward(&input_tensor, Some(&mut cache));

            let next_token_logits = logits.i((0, -1, ..));
            next_token = Sampler::sample(&next_token_logits, params, &tokens)?;

            if tx.blocking_send(next_token).is_err() {
                break; // Receiver dropped
            }
            tokens.push(next_token);

            // No hard max_seq_len cutoff — the evicting cache gracefully handles
            // unbounded generation by evicting middle tokens.
        }

        Ok(())
    }

    pub fn generate_batch(&mut self, requests: &[crate::batcher::Request]) -> Vec<Vec<i64>> {
        if requests.is_empty() {
            return vec![];
        }

        let batch_size = requests.len();
        let max_input_len = requests.iter().map(|r| r.input_ids.len()).max().unwrap_or(0);
        let max_tokens = requests.iter().map(|r| r.max_tokens).max().unwrap_or(10);

        let pad_id = 0;

        // 1. Prepare padded 2D input matrix
        let mut padded_inputs = vec![];
        for req in requests {
            let mut ids = req.input_ids.clone();
            ids.resize(max_input_len, pad_id);
            padded_inputs.extend_from_slice(&ids);
        }

        let mut current_tokens: Vec<Vec<i64>> = requests.iter().map(|r| r.input_ids.clone()).collect();
        let mut finished = vec![false; batch_size];

        // 2. Initialize evicting KV cache for the batch
        let mut cache = self.create_cache(batch_size as i64);

        let mut next_token_input = Tensor::from_slice(&padded_inputs)
            .view([batch_size as i64, max_input_len as i64])
            .to(self.device);

        for _step in 0..max_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            let logits = self.model.forward(&next_token_input, Some(&mut cache));

            let mut next_tokens = vec![];
            for b in 0..batch_size {
                if finished[b] {
                    next_tokens.push(pad_id);
                    continue;
                }

                let b_logits = logits.i((b as i64, -1, ..));
                let next = Sampler::sample(
                    &b_logits,
                    &SamplingParams::default(),
                    &current_tokens[b],
                )
                .unwrap_or(pad_id);
                next_tokens.push(next);

                current_tokens[b].push(next);
                // Use kv_cache_capacity as the effective limit for batched generation
                if current_tokens[b].len() >= self.model.config.kv_cache_capacity {
                    finished[b] = true;
                }
            }

            next_token_input = Tensor::from_slice(&next_tokens)
                .view([batch_size as i64, 1])
                .to(self.device);
        }

        current_tokens
    }
}

unsafe impl Send for Generator {}
