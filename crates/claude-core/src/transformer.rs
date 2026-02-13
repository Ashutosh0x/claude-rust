use tch::{nn, Tensor};
use crate::config::ModelConfig;
use crate::attention::CausalSelfAttention;
use crate::layer_norm::RMSNorm;

/// FeedForward block (MLP)
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
        x.apply(&self.c_fc).gelu("none").apply(&self.c_proj).dropout(self.dropout, true)
    }
}

unsafe impl Send for MLP {}
unsafe impl Sync for MLP {}


/// Transformer Block
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

    pub fn forward(&self, x: &Tensor, cache: Option<&mut crate::kv_cache::KVCache>) -> Tensor {
        let residual = x;
        let x_ln = self.ln_1.forward(x);
        
        let attn_out = self.attn.forward(&x_ln, cache);
        
        let x = residual + attn_out;
        
        let residual = &x;
        let x_ln = self.ln_2.forward(&x);
        let mlp_out = self.mlp.forward(&x_ln);
        
        residual + mlp_out
    }
}

unsafe impl Send for Block {}
unsafe impl Sync for Block {}


/// Full GPT Model
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
        let wte = nn::embedding(vs / "wte", config.vocab_size, config.n_embd, Default::default());
        let drop = config.dropout;
        
        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(Block::new(&(vs / "h" / i), config));
        }

        let ln_f = RMSNorm::new(&(vs / "ln_f"), config);
        let lm_head = nn::linear(vs / "lm_head", config.n_embd, config.vocab_size, nn::LinearConfig { bias: false, ..Default::default() });

        Self {
            wte,
            drop,
            blocks,
            ln_f,
            lm_head,
            config: config.clone(),
        }
    }

    /// past_key_values: Optional mutable slice of KVCache objects, one per layer.
    /// Returns: logits tensor
    pub fn forward(&self, idx: &Tensor, mut caches: Option<&mut [crate::kv_cache::KVCache]>) -> Tensor {
        let tok_emb = idx.apply(&self.wte); 
        let mut x = tok_emb.dropout(self.drop, true);
        
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = match caches {
                Some(ref mut c) => Some(&mut c[i]),
                None => None,
            };
            
            x = block.forward(&x, layer_cache);
        }

        x = self.ln_f.forward(&x); 
        let logits = x.apply(&self.lm_head);
        
        logits
    }
}

unsafe impl Send for ClaudeTransformer {}
unsafe impl Sync for ClaudeTransformer {}

