#[cfg(test)]
mod tests {
    use crate::config::ModelConfig;
    use crate::kv_cache::EvictingKVCache;
    use crate::rotary::RotaryEmbedding;
    use tch::{Device, Kind, Tensor};

    // =========================================================================
    // NTK RoPE Scaling Tests
    // =========================================================================

    #[test]
    fn test_ntk_scaling_no_extension() {
        // When max_seq_len <= original_max_seq_len, base should be unchanged
        let config = ModelConfig {
            max_seq_len: 4096,
            original_max_seq_len: 4096,
            rope_base: 10000.0,
            n_embd: 768,
            n_head: 12,
            ..Default::default()
        };
        let scaled = config.ntk_scaled_rope_base();
        assert!(
            (scaled - 10000.0).abs() < 1e-3,
            "Base should be unchanged when no extension needed, got {}",
            scaled
        );
    }

    #[test]
    fn test_ntk_scaling_with_extension() {
        // When max_seq_len > original_max_seq_len, base should increase
        let config = ModelConfig {
            max_seq_len: 1_000_000,
            original_max_seq_len: 4096,
            rope_base: 10000.0,
            n_embd: 4096,
            n_head: 32, // head_dim = 128
            ..Default::default()
        };
        let scaled = config.ntk_scaled_rope_base();
        // scale = 1M / 4K = ~244.14
        // head_dim = 128, factor = 128/(128-2) = 128/126 ≈ 1.0159
        // scaled_base = 10000 * 244.14^1.0159 ≈ much higher than 10000
        assert!(
            scaled > 10000.0,
            "NTK scaling should increase the base, got {}",
            scaled
        );
        assert!(
            scaled > 2_000_000.0,
            "With 250x scale, base should be significantly higher, got {}",
            scaled
        );
    }

    #[test]
    fn test_rotary_embedding_shape() {
        let config = ModelConfig {
            max_seq_len: 128,
            n_embd: 64,
            n_head: 4, // head_dim = 16
            ..Default::default()
        };
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(&config, device);

        // Apply to a test tensor: [batch=1, heads=4, seq=8, head_dim=16]
        let x = Tensor::randn(&[1, 4, 8, 16], (Kind::Float, device));
        let result = rope.apply_with_offset(&x, 0);

        assert_eq!(result.size(), vec![1, 4, 8, 16], "Output shape should match input");
    }

    #[test]
    fn test_rotary_position_offset_differs() {
        let config = ModelConfig {
            max_seq_len: 256,
            n_embd: 64,
            n_head: 4,
            ..Default::default()
        };
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(&config, device);

        let x = Tensor::ones(&[1, 4, 1, 16], (Kind::Float, device));

        let r0 = rope.apply_with_offset(&x, 0);
        let r10 = rope.apply_with_offset(&x, 10);

        // Different positions should give different embeddings
        let diff = (&r0 - &r10).abs().sum(Kind::Float);
        assert!(
            f64::try_from(&diff).unwrap() > 0.0,
            "Different position offsets should produce different embeddings"
        );
    }

    // =========================================================================
    // Evicting KV Cache Tests
    // =========================================================================

    #[test]
    fn test_cache_basic_append() {
        let device = Device::Cpu;
        let mut cache = EvictingKVCache::new(
            1,  // n_layers
            1,  // batch
            2,  // n_heads
            4,  // head_dim
            16, // max_capacity
            2,  // sink_size
            device,
        );

        let k = Tensor::ones(&[1, 2, 3, 4], (Kind::Float, device));
        let v = Tensor::ones(&[1, 2, 3, 4], (Kind::Float, device)) * 2.0;

        cache.append(0, &k, &v);

        assert_eq!(cache.current_len, 3);
        assert_eq!(cache.total_tokens_seen, 3);

        let (kv, vv) = cache.get_view(0);
        assert_eq!(kv.size(), vec![1, 2, 3, 4]);
        assert_eq!(vv.size(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_cache_eviction_preserves_sinks() {
        let device = Device::Cpu;
        let sink_size = 2;
        let max_capacity = 8;

        let mut cache = EvictingKVCache::new(
            1,            // n_layers
            1,            // batch
            1,            // n_heads
            1,            // head_dim (1 for easy inspection)
            max_capacity,
            sink_size,
            device,
        );

        // Fill cache to capacity: 8 tokens
        for i in 0..8i64 {
            let k = Tensor::from_slice(&[i as f32]).view([1, 1, 1, 1]);
            let v = Tensor::from_slice(&[i as f32]).view([1, 1, 1, 1]);
            cache.append(0, &k, &v);
        }
        assert_eq!(cache.current_len, 8);

        // Append 2 more — should trigger eviction
        let k_new = Tensor::from_slice(&[100.0f32, 101.0]).view([1, 1, 2, 1]);
        let v_new = Tensor::from_slice(&[100.0f32, 101.0]).view([1, 1, 2, 1]);
        cache.append(0, &k_new, &v_new);

        assert_eq!(cache.current_len, max_capacity);
        assert_eq!(cache.total_tokens_seen, 10);

        // Verify sink tokens (positions 0, 1) are preserved
        let (k_view, _) = cache.get_view(0);
        let k_flat: Vec<f32> = Vec::try_from(k_view.flatten(0, -1)).unwrap();

        // First two should be original sinks (0.0, 1.0)
        assert!(
            (k_flat[0] - 0.0).abs() < 1e-6,
            "Sink token 0 should be preserved, got {}",
            k_flat[0]
        );
        assert!(
            (k_flat[1] - 1.0).abs() < 1e-6,
            "Sink token 1 should be preserved, got {}",
            k_flat[1]
        );

        // Last two should be the newly appended tokens (100.0, 101.0)
        let last = k_flat.len();
        assert!(
            (k_flat[last - 2] - 100.0).abs() < 1e-6,
            "New token should be at end, got {}",
            k_flat[last - 2]
        );
        assert!(
            (k_flat[last - 1] - 101.0).abs() < 1e-6,
            "New token should be at end, got {}",
            k_flat[last - 1]
        );
    }

    #[test]
    fn test_cache_clear() {
        let device = Device::Cpu;
        let mut cache = EvictingKVCache::new(1, 1, 1, 4, 16, 2, device);

        let k = Tensor::ones(&[1, 1, 5, 4], (Kind::Float, device));
        let v = Tensor::ones(&[1, 1, 5, 4], (Kind::Float, device));
        cache.append(0, &k, &v);
        assert_eq!(cache.current_len, 5);

        cache.clear();
        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.total_tokens_seen, 0);
    }

    // =========================================================================
    // Config Serialization Tests
    // =========================================================================

    #[test]
    fn test_config_backward_compatible_deserialization() {
        // Old config without new fields should deserialize with defaults
        let yaml = r#"
n_embd: 768
n_head: 12
n_layer: 12
vocab_size: 50257
max_seq_len: 2048
dropout: 0.1
layer_norm_epsilon: 0.00001
use_bias: false
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).expect("Should deserialize old config");
        assert_eq!(config.original_max_seq_len, 4096);
        assert_eq!(config.window_size, 4096);
        assert_eq!(config.sink_tokens, 4);
        assert_eq!(config.kv_cache_capacity, 32768);
        assert!((config.rope_base - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_config_with_new_fields() {
        let yaml = r#"
n_embd: 4096
n_head: 32
n_layer: 32
vocab_size: 50257
max_seq_len: 1000000
original_max_seq_len: 4096
dropout: 0.0
layer_norm_epsilon: 0.00001
use_bias: false
rope_base: 10000.0
window_size: 4096
sink_tokens: 4
kv_cache_capacity: 32768
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).expect("Should deserialize full config");
        assert_eq!(config.max_seq_len, 1_000_000);
        assert_eq!(config.original_max_seq_len, 4096);
        assert_eq!(config.kv_cache_capacity, 32768);
    }
}
