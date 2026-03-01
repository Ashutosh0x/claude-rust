//! Standalone transformer block (alternative grouping).
//!
//! This module re-exports the Block from transformer.rs for convenience.
//! The actual implementation lives in transformer.rs where it's integrated
//! with the full ClaudeTransformer pipeline.

pub use crate::transformer::Block;
