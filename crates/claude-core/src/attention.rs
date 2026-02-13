use tch::{nn, Tensor, Kind, IndexOp};
use crate::config::ModelConfig;
use crate::rotary::RotaryEmbedding;

pub struct CausalSelfAttention {
    c_attn: nn::Linear,
    c_proj: nn::Linear,
    n_head: i64,
    dropout: f64,
    bias: Tensor,
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
        
        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(head_dim, vs.device()));

        // Causal mask
        let mask = Tensor::ones(&[config.max_seq_len, config.max_seq_len], (Kind::Bool, vs.device()))
            .tril(0)
            .reshape(&[1, 1, config.max_seq_len, config.max_seq_len]);

        Self {
            c_attn,
            c_proj,
            n_head,
            dropout: config.dropout,
            bias: mask.to_kind(Kind::Float),
            rotary_emb,
        }
    }

    pub fn forward(&self, x: &Tensor, cache: Option<&mut crate::kv_cache::KVCache>) -> Tensor {
        let (b, t, c) = x.size3().unwrap(); 
        
        let qkv = x.apply(&self.c_attn);
        let chunks = qkv.chunk(3, -1);
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);
        
        let head_size = c / self.n_head;
        
        let mut k = k.view([b, t, self.n_head, head_size]).transpose(1, 2);
        let mut q = q.view([b, t, self.n_head, head_size]).transpose(1, 2);
        let v = v.view([b, t, self.n_head, head_size]).transpose(1, 2);

        // Apply RoPE
        let past_len = match cache {
            Some(ref c) => c.length as i64,
            None => 0,
        };
        
        q = self.rotary_emb.forward(&q, t + past_len).i((.., .., past_len.., ..));
        k = self.rotary_emb.forward(&k, t + past_len).i((.., .., past_len.., ..));

        // KV Cache handling
        let (k_full, v_full) = match cache {
            Some(c) => {
                c.update(&k, &v);
                c.get_view()
            },
            None => (k, v),
        };
        
        let att = q.matmul(&k_full.transpose(-2, -1)) * (1.0 / (head_size as f64).sqrt());
        
        let total_t = k_full.size()[2];
        
        // Apply mask only if we are the first step (past_len == 0) and T > 1
        if past_len == 0 && t > 1 {
             let mask = self.bias.i((.., .., ..total_t, ..total_t));
             let att = att.masked_fill(&mask.eq(0.0), f64::NEG_INFINITY);
             let att = att.softmax(-1, Kind::Float);
             let att = att.dropout(self.dropout, true);
             let y = att.matmul(&v_full);
             let y = y.transpose(1, 2).contiguous().view([b, t, c]);
             y.apply(&self.c_proj)
        } else {
             let att = att.softmax(-1, Kind::Float);
             let att = att.dropout(self.dropout, true);
             let y = att.matmul(&v_full);
             let y = y.transpose(1, 2).contiguous().view([b, t, c]);
             y.apply(&self.c_proj)
        }
    }
}
