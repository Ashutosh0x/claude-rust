use crate::sampling::{Sampler, SamplingParams};
use claude_core::ClaudeTransformer;
use tch::{Device, IndexOp, Tensor};

use std::sync::Arc;

pub struct Generator {
    model: Arc<ClaudeTransformer>,
    device: Device,
}

impl Generator {
    pub fn new(model: Arc<ClaudeTransformer>, device: Device) -> Self {
        Self { model, device }
    }

    pub fn generate_stream(
        &mut self,
        prompt_ids: &[i64],
        max_new_tokens: usize,
        params: &SamplingParams,
        tx: tokio::sync::mpsc::Sender<i64>,
    ) -> anyhow::Result<()> {
        if prompt_ids.is_empty() {
            return Ok(());
        }

        let max_seq_len = self.model.config.max_seq_len as usize;
        let mut tokens = if prompt_ids.len() > max_seq_len {
            prompt_ids[prompt_ids.len() - max_seq_len..].to_vec()
        } else {
            prompt_ids.to_vec()
        };

        // Initialize KV Caches for each layer
        let mut caches: Vec<claude_core::kv_cache::KVCache> = (0..self.model.config.n_layer)
            .map(|_| {
                claude_core::kv_cache::KVCache::new(
                    self.model.config.max_seq_len as usize,
                    self.model.config.n_head,
                    self.model.config.n_embd / self.model.config.n_head,
                    self.device,
                    tch::Kind::Float,
                )
            })
            .collect();

        // 1. Prefill
        let input_tensor = Tensor::from_slice(&tokens)
            .view([1, tokens.len() as i64])
            .to(self.device);
        let logits = self.model.forward(&input_tensor, Some(&mut caches));

        // Sample first new token
        let next_token_logits = logits.i((0, -1, ..));
        let mut next_token = Sampler::sample(&next_token_logits, params, &tokens)?;

        // Yield first token
        if tx.try_send(next_token).is_err() {
            return Ok(());
        }
        tokens.push(next_token);

        // 2. Decode Loop
        for _ in 0..max_new_tokens {
            let input_tensor = Tensor::from_slice(&[next_token])
                .view([1, 1])
                .to(self.device);
            let logits = self.model.forward(&input_tensor, Some(&mut caches));

            let next_token_logits = logits.i((0, -1, ..));
            next_token = Sampler::sample(&next_token_logits, params, &tokens)?;

            // Yield token
            if tx.try_send(next_token).is_err() {
                break; // Receiver dropped or channel is full
            }
            tokens.push(next_token);

            if tokens.len() >= max_seq_len {
                break;
            }
        }

        Ok(())
    }
}

unsafe impl Send for Generator {}
