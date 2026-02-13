# Production Roadmap & Engineering Backlog

This document tracks the technical debt, feature requests, and architectural milestones for the Claude-Rust project.

## Status: Development (Core Ready)
- **Crates**: All crates (`claude-core`, `inference`, `tokenizer`, `trainer`, `retrieval`, `claude-tui`) are compiling.
- **Inference**: SSE Streaming server is functional.
- **Tokenizer**: Custom BPE trainer and decoder are functional.

---

## Active Technical Backlog

### 1. High-Performance Core (`claude-core`)
- [ ] **Static KV Cache Allocation**:
  - Currently, KV cache grows via `Tensor::cat`, which is $O(N^2)$ in memory copies.
  - **Task**: Allocate a fixed-size buffer `[B, H, MaxSeq, D]` and use in-place updates.
- [ ] **Safetensors Integration**:
  - **Task**: Replace `tch` VarStore `.save/.load` (PyTorch format) with `safetensors` for zero-copy loading and cross-platform compatibility.
- [ ] **Grouped Query Attention (GQA)**:
  - **Task**: Update attention logic to support GQA (used by Llama-3 and Claude-like architectures) to reduce memory bandwidth.

### 2. Advanced Inference (`inference`)
- [ ] **Continuous Batching**:
  - **Task**: Refactor the `Generator` to handle a queue of requests, interleaving token generation to maximize GPU throughput.
- [ ] **Stopping Criteria & Logit Bias**:
  - **Task**: Add support for `stop_sequences` (e.g., `["\nUser:"]`) and logit biases in the `SamplingParams`.
- [ ] **Beam Search**:
  - **Task**: Implement beam search as an alternative to greedy/nucleus sampling for reasoning tasks.

### 3. Intelligence & RAG (`retrieval`)
- [ ] **Store Persistence**:
  - **Task**: Implement `.save()` and `.load()` for the `VectorStore` using `serde` and `safetensors`.
- [ ] **RAG Integration Hook**:
  - **Task**: Add a "Context Provider" trait to the generation loop that injecting retrieved snippets into the prompt dynamically.
- [ ] **Embedding Pipeline**:
  - **Task**: Integration of a small BERT or similar model for generating the embeddings used in `retrieval`.

### 4. Efficiency & Quantization (`quant`)
- [ ] **INT8 Weight Quantization**:
  - **Task**: Implement symmetric/asymmetric linear quantization for model weights.
- [ ] **KV Cache Quantization**:
  - **Task**: Cache keys and values in FP8 or INT8 to double the effective context length.

### 5. UI/UX Polish (`claude-tui`)
- [ ] **Markdown Rendering**:
  - **Task**: Use a crate like `pulldown-cmark` or `ratatui-markdown` to render formatted text in the chat window.
- [ ] **Horizontal Scrolling & Code Blocks**:
  - **Task**: Implement code block detection and syntax highlighting in the TUI using `syntect`.

---

## Execution Plan

### Immediate Next: KV Cache Optimization
The `Tensor::cat` in `CausalSelfAttention::forward` will cause severe slowdowns as the context reaches 100+ tokens. 

**Plan**: 
1. Modify `CausalSelfAttention` to hold a pre-allocated `Option<Tensor>` for KV.
2. Update the forward pass to use `.narrow_copy()` or index assignments.
3. Update the `Generator` to manage the cache state more explicitly.
