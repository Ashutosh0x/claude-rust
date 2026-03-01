//! Text chunking utilities for RAG pipelines.
//!
//! Splits documents into overlapping chunks for embedding and retrieval.

/// Configuration for text chunking.
pub struct ChunkerConfig {
    /// Maximum number of characters per chunk.
    pub chunk_size: usize,
    /// Number of characters to overlap between adjacent chunks.
    pub overlap: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
        }
    }
}

/// A chunk of text with position metadata.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub chunk_index: usize,
}

/// Split text into overlapping chunks at sentence boundaries.
pub fn chunk_text(text: &str, config: &ChunkerConfig) -> Vec<TextChunk> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < text.len() {
        let mut end = (start + config.chunk_size).min(text.len());

        // Try to break at a sentence boundary (. ! ? \n)
        if end < text.len() {
            if let Some(boundary) = text[start..end]
                .rfind(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
            {
                end = start + boundary + 1;
            }
        }

        let chunk_text = text[start..end].trim().to_string();
        if !chunk_text.is_empty() {
            chunks.push(TextChunk {
                text: chunk_text,
                start_char: start,
                end_char: end,
                chunk_index,
            });
            chunk_index += 1;
        }

        // Advance with overlap
        start = if end > config.overlap {
            end - config.overlap
        } else {
            end
        };

        // Prevent infinite loop on very small text
        if start >= end {
            break;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking() {
        let text = "Hello world. This is a test. Another sentence here. And more.";
        let config = ChunkerConfig {
            chunk_size: 30,
            overlap: 5,
        };
        let chunks = chunk_text(text, &config);
        assert!(!chunks.is_empty());
        // Every chunk should be non-empty
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }
}
