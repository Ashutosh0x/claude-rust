//! Byte-level tokenizer fallback.
//!
//! Provides byte-level encoding/decoding for handling unknown characters
//! outside the BPE vocabulary (similar to GPT-2's byte fallback).

/// Encode a string as raw bytes (one token per byte).
pub fn byte_encode(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

/// Decode raw byte tokens back to a string.
pub fn byte_decode(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .filter_map(|&t| if t <= 255 { Some(t as u8) } else { None })
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

/// Generate byte-level vocabulary entries (tokens 0-255).
pub fn byte_vocab() -> Vec<(String, u32)> {
    (0u32..=255)
        .map(|b| {
            let label = if b >= 33 && b <= 126 {
                format!("{}", b as u8 as char)
            } else {
                format!("<0x{:02X}>", b)
            };
            (label, b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_roundtrip() {
        let text = "Hello, World!";
        let encoded = byte_encode(text);
        let decoded = byte_decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_byte_vocab_size() {
        assert_eq!(byte_vocab().len(), 256);
    }
}
