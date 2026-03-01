//! Tokenizer interface traits.
//!
//! Defines the common interface that all tokenizer implementations must support.

/// Core tokenizer trait.
pub trait Tokenize {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode token IDs back to text.
    fn decode(&self, ids: &[u32]) -> String;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Tokenizer with max-token-limited encoding.
pub trait TokenizeWithLimit: Tokenize {
    /// Encode text, truncating to at most `max_tokens`.
    fn encode_with_max_tokens(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        let mut tokens = self.encode(text);
        tokens.truncate(max_tokens);
        tokens
    }
}
