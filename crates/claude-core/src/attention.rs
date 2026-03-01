use tch::{nn, Tensor, Kind};
use crate::config::ModelConfig;
use crate::rotary::RotaryEmbedding;
use crate::kv_cache::EvictingKVCache;

/// Sliding Window Causal Self-Attention with Sink Tokens.
///
/// Instead of full O(N²) attention, each token attends to:
/// 1. The first `sink_tokens` positions (global context anchors)
/// 2. The last `window_size` positions before it (local sliding window)
///
/// This reduces attention complexity to O(N × (W + S)) where W = window_size
/// and S = sink_tokens.
pub struct CausalSelfAttention {
    c_attn: nn::Linear,
    c_proj: nn::Linear,
    n_head: i64,
    head_dim: i64,
    dropout: f64,
    window_size: i64,
    sink_tokens: i64,
    rotary_emb: std::sync::Arc<RotaryEmbedding>,
}

impl CausalSelfAttention {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let head_dim = n_embd / n_head;

        let linear_config = nn::LinearConfig {
            bias: config.use_bias,
            ..Default::default()
        };

        let c_attn = nn::linear(vs / "c_attn", n_embd, 3 * n_embd, linear_config);
        let c_proj = nn::linear(vs / "c_proj", n_embd, n_embd, linear_config);

        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(config, vs.device()));

        Self {
            c_attn,
            c_proj,
            n_head,
            head_dim,
            dropout: config.dropout,
            window_size: config.window_size,
            sink_tokens: config.sink_tokens,
            rotary_emb,
        }
    }

    /// Build a sliding window causal attention mask.
    ///
    /// For each query position i, the mask allows attending to:
    /// - Positions [0, sink_tokens) — always visible (global anchors)
    /// - Positions [max(sink_tokens, i - window_size + 1), i] — local window
    ///
    /// Returns a [seq_q, seq_kv] mask where 0.0 = attend, -inf = mask out.
    fn build_sliding_window_mask(
        &self,
        seq_q: i64,
        seq_kv: i64,
        offset: i64,
        device: tch::Device,
    ) -> Tensor {
        // Start with everything masked
        let mut mask = Tensor::full(
            &[seq_q, seq_kv],
            f64::NEG_INFINITY,
            (Kind::Float, device),
        );

        // For prefill (seq_q > 1), build per-row masks
        // For decode (seq_q == 1), the single query attends to everything in KV cache
        if seq_q == 1 {
            // Decode: single token attends to all cached KV positions
            // The sliding window constraint is enforced by the KV cache eviction,
            // not the mask — the cache only holds sink + recent window tokens.
            let _ = mask.fill_(0.0);
        } else {
            // Prefill: build sliding window + sink causal mask
            for q_idx in 0..seq_q {
                let abs_pos = offset + q_idx; // absolute position in the full sequence

                // Sink tokens: always attend to [0, sink_tokens)
                let sink_end = self.sink_tokens.min(seq_kv);
                if sink_end > 0 {
                    let _ = mask.narrow(0, q_idx, 1)
                        .narrow(1, 0, sink_end)
                        .fill_(0.0);
                }

                // Local window: attend to [max(sink_tokens, abs_pos - window + 1), abs_pos]
                // mapped to KV indices
                let window_start_abs = (abs_pos - self.window_size + 1).max(self.sink_tokens);
                let window_end_abs = abs_pos; // inclusive

                // Map absolute positions to KV tensor indices
                // During prefill, KV positions [0..seq_kv) map to absolute [offset..offset+seq_kv)
                // But we also need to handle the case where KV includes cached tokens
                let kv_start = (window_start_abs - offset).max(self.sink_tokens).min(seq_kv);
                let kv_end = (window_end_abs - offset + 1).min(seq_kv);

                if kv_end > kv_start {
                    let _ = mask.narrow(0, q_idx, 1)
                        .narrow(1, kv_start, kv_end - kv_start)
                        .fill_(0.0);
                }
            }
        }

        mask
    }

    /// Forward pass with evicting KV cache support.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_embd]
    /// * `cache` - Optional evicting KV cache (for inference)
    /// * `layer_idx` - Layer index into the cache
    /// * `position_offset` - Absolute position of the first token in `x`
    pub fn forward(
        &self,
        x: &Tensor,
        cache: Option<&mut EvictingKVCache>,
        layer_idx: usize,
        position_offset: i64,
    ) -> Tensor {
        let (b, t, c) = x.size3().unwrap();
        let device = x.device();

        // Project Q, K, V
        let qkv = x.apply(&self.c_attn);
        let chunks = qkv.chunk(3, -1);
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);

        // Reshape: [b, t, n_head, head_dim] -> [b, n_head, t, head_dim]
        let q = q.view([b, t, self.n_head, self.head_dim]).transpose(1, 2);
        let k = k.view([b, t, self.n_head, self.head_dim]).transpose(1, 2);
        let v = v.view([b, t, self.n_head, self.head_dim]).transpose(1, 2);

        // Apply RoPE with correct absolute positions
        let q = self.rotary_emb.apply_with_offset(&q, position_offset);
        let k = self.rotary_emb.apply_with_offset(&k, position_offset);

        // KV Cache handling
        let (k_full, v_full) = match cache {
            Some(c) => {
                c.append(layer_idx, &k, &v);
                c.get_view(layer_idx)
            }
            None => (k, v),
        };

        let total_kv = k_full.size()[2];

        // Compute attention scores
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k_full.transpose(-2, -1)) / scale;

        // Apply sliding window mask
        let mask = self.build_sliding_window_mask(t, total_kv, position_offset, device);
        let scores = scores + mask.unsqueeze(0).unsqueeze(0); // [1, 1, seq_q, seq_kv]

        let weights = scores.softmax(-1, Kind::Float);
        let weights = weights.dropout(self.dropout, true);

        // Weighted sum of values
        let y = weights.matmul(&v_full); // [b, n_head, t, head_dim]
        let y = y.transpose(1, 2).contiguous().view([b, t, c]);

        y.apply(&self.c_proj)
    }
}
