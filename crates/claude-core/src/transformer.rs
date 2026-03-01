use tch::{nn, Tensor};
use crate::config::ModelConfig;
use crate::attention::CausalSelfAttention;
use crate::layer_norm::RMSNorm;
use crate::kv_cache::EvictingKVCache;

/// FeedForward block (SwiGLU MLP)
pub struct MLP {
    c_fc: nn::Linear,
    c_proj: nn::Linear,
    dropout: f64,
}

impl MLP {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let n_embd = config.n_embd;
        let n_hidden = 4 * n_embd;

        let c_fc = nn::linear(vs / "c_fc", n_embd, n_hidden, Default::default());
        let c_proj = nn::linear(vs / "c_proj", n_hidden, n_embd, Default::default());

        Self {
            c_fc,
            c_proj,
            dropout: config.dropout,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.apply(&self.c_fc)
            .gelu("none")
            .apply(&self.c_proj)
            .dropout(self.dropout, true)
    }
}

unsafe impl Send for MLP {}
unsafe impl Sync for MLP {}


/// Transformer Block with sliding window attention and evicting KV cache support.
pub struct Block {
    ln_1: RMSNorm,
    attn: CausalSelfAttention,
    ln_2: RMSNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let ln_1 = RMSNorm::new(&(vs / "ln_1"), config);
        let attn = CausalSelfAttention::new(&(vs / "attn"), config);
        let ln_2 = RMSNorm::new(&(vs / "ln_2"), config);
        let mlp = MLP::new(&(vs / "mlp"), config);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    /// Forward pass with evicting KV cache and position offset.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `cache` - Optional evicting KV cache for inference
    /// * `layer_idx` - This block's layer index (for cache indexing)
    /// * `position_offset` - Absolute position of the first token in x
    pub fn forward(
        &self,
        x: &Tensor,
        cache: Option<&mut EvictingKVCache>,
        layer_idx: usize,
        position_offset: i64,
    ) -> Tensor {
        // Pre-norm + attention + residual
        let residual = x;
        let x_ln = self.ln_1.forward(x);
        let attn_out = self.attn.forward(&x_ln, cache, layer_idx, position_offset);
        let x = residual + attn_out;

        // Pre-norm + FFN + residual
        let residual = &x;
        let x_ln = self.ln_2.forward(&x);
        let mlp_out = self.mlp.forward(&x_ln);

        residual + mlp_out
    }
}

unsafe impl Send for Block {}
unsafe impl Sync for Block {}


/// Full Transformer Model with long-context support.
///
/// Supports two modes:
/// 1. **Training / no cache**: `forward(idx, None)` — standard causal attention
/// 2. **Inference with cache**: `forward(idx, Some(cache))` — uses evicting KV cache
///    with sliding window attention and NTK-aware RoPE scaling
pub struct ClaudeTransformer {
    wte: nn::Embedding,
    drop: f64,
    blocks: Vec<Block>,
    ln_f: RMSNorm,
    lm_head: nn::Linear,
    pub config: ModelConfig,
}

impl ClaudeTransformer {
    pub fn new(vs: &nn::Path, config: &ModelConfig) -> Self {
        let wte = nn::embedding(
            vs / "wte",
            config.vocab_size,
            config.n_embd,
            Default::default(),
        );
        let drop = config.dropout;

        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(Block::new(&(vs / "h" / i), config));
        }

        let ln_f = RMSNorm::new(&(vs / "ln_f"), config);
        let lm_head = nn::linear(
            vs / "lm_head",
            config.n_embd,
            config.vocab_size,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            wte,
            drop,
            blocks,
            ln_f,
            lm_head,
            config: config.clone(),
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `idx` - Token indices [batch, seq_len]
    /// * `cache` - Optional evicting KV cache. When provided, the cache's
    ///   `total_tokens_seen` is used as the position offset for RoPE, and
    ///   each layer appends its KV pairs to the cache.
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, vocab_size]
    pub fn forward(
        &self,
        idx: &Tensor,
        mut cache: Option<&mut EvictingKVCache>,
    ) -> Tensor {
        // Compute position offset from cache state
        // The offset is total_tokens_seen BEFORE appending the current tokens,
        // since total_tokens_seen gets incremented inside append.
        // But since we read it before forward, this is the correct value.
        let position_offset = match &cache {
            Some(c) => c.total_tokens_seen as i64,
            None => 0,
        };

        let tok_emb = idx.apply(&self.wte);
        let mut x = tok_emb.dropout(self.drop, true);

        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = match cache {
                Some(ref mut c) => Some(&mut **c),
                None => None,
            };
            x = block.forward(&x, layer_cache, i, position_offset);
        }

        x = self.ln_f.forward(&x);
        x.apply(&self.lm_head)
    }
}

unsafe impl Send for ClaudeTransformer {}
unsafe impl Sync for ClaudeTransformer {}
