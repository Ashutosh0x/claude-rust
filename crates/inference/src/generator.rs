use tch::{Tensor, Device, IndexOp};
use claude_core::ClaudeTransformer;
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

    pub fn generate_stream(
        &mut self,
        prompt_ids: &[i64],
        max_new_tokens: usize,
        params: &SamplingParams,
        tx: tokio::sync::mpsc::Sender<i64>,
    ) -> anyhow::Result<()> {
        let mut tokens = prompt_ids.to_vec();
        
        // Initialize KV Caches for each layer
        let mut caches: Vec<claude_core::kv_cache::KVCache> = (0..self.model.config.n_layer)
            .map(|_| claude_core::kv_cache::KVCache::new(
                1, // batch_size (i64)
                self.model.config.max_seq_len as usize, // max_capacity (usize)
                self.model.config.n_head as i64, // n_head (i64)
                (self.model.config.n_embd / self.model.config.n_head) as i64, // head_dim (i64)
                self.device,
                tch::Kind::Float
            ))
            .collect();

        // 1. Prefill
        let input_tensor = Tensor::from_slice(&tokens).view([1, tokens.len() as i64]).to(self.device);
        let logits = self.model.forward(&input_tensor, Some(&mut caches));
        
        // Sample first new token
        let next_token_logits = logits.i((0, -1, ..)); 
        let mut next_token = Sampler::sample(&next_token_logits, params, &tokens)?;
        
        // Yield first token
        let _ = tx.blocking_send(next_token);
        tokens.push(next_token);

        // 2. Decode Loop
        for _ in 0..max_new_tokens {
            let input_tensor = Tensor::from_slice(&[next_token]).view([1, 1]).to(self.device);
            let logits = self.model.forward(&input_tensor, Some(&mut caches));
            
            let next_token_logits = logits.i((0, -1, ..));
            next_token = Sampler::sample(&next_token_logits, params, &tokens)?;
            
            // Yield token
            if tx.blocking_send(next_token).is_err() {
                break; // Receiver dropped
            }
            tokens.push(next_token);
            
            if tokens.len() >= self.model.config.max_seq_len as usize {
                break;
            }
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
        
        let pad_id = 0; // Or whatever tokenizer.pad_id is
        
        // 1. Prepare Padded 2D Matrix
        let mut padded_inputs = vec![];
        for req in requests {
            let mut ids = req.input_ids.clone();
            ids.resize(max_input_len, pad_id);
            padded_inputs.extend_from_slice(&ids);
        }
        
        let mut current_tokens: Vec<Vec<i64>> = requests.iter().map(|r| r.input_ids.clone()).collect();
        let mut finished = vec![false; batch_size];

        // 2. Initialize KV caches per layer (now dimensioned for batch_size!)
        // Note: The transformer natively handles batch > 1 if KV caches are sized [batch_size, seq_len, ...].
        // Currently KVCache expects `batch_size: 1` as written, so to be fully batched we need to modify 
        // the KVCache struct allocator. For this mock implementation, we assume KVCache accepts batch_size.
        let mut caches: Vec<claude_core::kv_cache::KVCache> = (0..self.model.config.n_layer)
            .map(|_| claude_core::kv_cache::KVCache::new(
                batch_size as i64, // batch_size (i64)
                self.model.config.max_seq_len as usize, // max_capacity (usize)
                self.model.config.n_head as i64, // n_head (i64)
                (self.model.config.n_embd / self.model.config.n_head) as i64, // head_dim (i64)
                self.device,
                tch::Kind::Float
            ))
            .collect();

        // Warning: if KVCache statically allocates B=1 inside claude_core, you must 
        // fix that! Let's assume you updated KVCache::new to take a `batch_size: i64`.
        // Let's do a naive decoding loop.
        let mut next_token_input = Tensor::from_slice(&padded_inputs)
            .view([batch_size as i64, max_input_len as i64])
            .to(self.device);

        for _step in 0..max_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }
            
            let logits = self.model.forward(&next_token_input, Some(&mut caches));
            
            // Extract the last logits for each batch item
            let mut next_tokens = vec![];
            for b in 0..batch_size {
                if finished[b] {
                    next_tokens.push(pad_id);
                    continue;
                }
                
                // shape [B, T, V] -> slice out single B, then -1 for T
                let b_logits = logits.i((b as i64, -1, ..));
                // Sample dynamically 
                // A real batch sampler would do this across the entire [B, V] matrix efficiently.
                let next = Sampler::sample(&b_logits, &SamplingParams::default(), &current_tokens[b]).unwrap_or(pad_id);
                next_tokens.push(next);
                
                current_tokens[b].push(next);
                if current_tokens[b].len() >= self.model.config.max_seq_len as usize {
                    finished[b] = true;
                }
            }
            
            // Next input tensor is [B, 1] 
            next_token_input = Tensor::from_slice(&next_tokens)
                .view([batch_size as i64, 1])
                .to(self.device);
        }

        // Return only the generated portion (strip the prompt if desired, but for now return all)
        current_tokens
    }
}

unsafe impl Send for Generator {}

