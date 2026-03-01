//! Tokenizer unit tests.

#[cfg(test)]
mod byte_tokenizer_tests {
    use crate::byte_tokenizer::{byte_encode, byte_decode, byte_vocab};

    #[test]
    fn test_ascii_roundtrip() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let tokens = byte_encode(text);
        let decoded = byte_decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_empty_string() {
        assert!(byte_encode("").is_empty());
        assert_eq!(byte_decode(&[]), "");
    }

    #[test]
    fn test_vocab_coverage() {
        let vocab = byte_vocab();
        assert_eq!(vocab.len(), 256);
        // Check that IDs are 0-255
        for (i, (_, id)) in vocab.iter().enumerate() {
            assert_eq!(*id, i as u32);
        }
    }
}
