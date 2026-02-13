use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("Vocabulary mismatch")]
    VocabMismatch,

    #[error("Token not found: {0}")]
    TokenNotFound(String),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
