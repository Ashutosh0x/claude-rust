# Production Roadmap & Engineering Backlog

This document tracks the technical debt, feature requests, and architectural milestones for the Claude-Rust project.

## Status: Development (Feature-Complete Core)
- **Crates**: All 11 crates (`claude-core`, `inference`, `tokenizer`, `trainer`, `retrieval`, `quant`, `tensors`, `agent`, `claude-tui`, `utils`, tools) are compiling with zero warnings.
- **Inference**: SSE Streaming server is functional with long-context support (up to 1M tokens).
- **Long-Context**: NTK-Aware RoPE, Sliding Window Attention, Evicting KV Cache, and Sink Tokens are implemented.
- **Training**: Full pipeline with checkpoint save/resume, cosine LR scheduler, JSONL metrics.
- **Quantization**: INT8 and Q4 per-group quantization for model compression.
- **RAG**: Text chunking, embedding, flat cosine similarity index.
- **Tokenizer**: Custom BPE trainer and decoder with byte-level fallback.

---

## Completed

### ✅ Long-Context Architecture
- [x] **NTK-Aware RoPE Scaling**: Extends positional embeddings beyond training length (4K → 1M) via NTK-scaled base frequency.
- [x] **Sliding Window Attention**: O(N × W) local attention with global sink token support, replacing O(N²) full attention.
- [x] **Evicting KV Cache**: Pre-allocated ring buffer with sink pinning — bounded memory, graceful middle-token eviction.
- [x] **Attention Sink Tokens**: First N tokens pinned as global context anchors for stable attention quality.
- [x] **Static KV Cache Allocation**: Replaced `Tensor::cat` O(N²) copies with pre-allocated in-place writes.

### ✅ Quantization (`quant`)
- [x] **INT8 Weight Quantization**: Symmetric per-tensor quantization with roundtrip accuracy (4x compression).
- [x] **Q4 Weight Quantization**: Per-group 4-bit packing with two nibbles per byte (~8x compression).
- [x] **Calibration Stats**: Symmetric and asymmetric calibration utilities.

### ✅ Training Pipeline (`trainer`)
- [x] **Checkpoint Manager**: Save/load/resume with step metadata and model config.
- [x] **Cosine LR Scheduler**: Linear warmup + cosine decay.
- [x] **Metrics Logger**: JSONL output with loss, perplexity, LR, throughput.
- [x] **Streaming DataLoader**: Binary token file reader with sequential batching.
- [x] **Full Training Loop**: Wires optimizer, scheduler, metrics, and checkpointing.

### ✅ RAG Pipeline (`retrieval`)
- [x] **Text Chunker**: Sentence-boundary splitting with configurable overlap.
- [x] **Embedder Trait**: `MeanPoolEmbedder` using token embedding table.
- [x] **Flat Index**: Brute-force cosine similarity search.
- [x] **FAISS Config**: IVF/PQ factory string generation stub.

### ✅ Infrastructure
- [x] **Tensors Abstraction**: Backend trait, tensor ops, CUDA utilities.
- [x] **Utils**: Config load/save, filesystem helpers, logging init.
- [x] **Core Modules**: Embedding layer, LM head, weight init, sinusoidal encoding.

---

## Active Technical Backlog

### 1. High-Performance Core (`claude-core`)
- [ ] **Grouped Query Attention (GQA)**:
  - **Task**: Update attention logic to support GQA (used by Llama-3 and Claude-like architectures) to reduce memory bandwidth.
- [ ] **KV Cache Quantization**:
  - **Task**: Cache keys and values in FP8 or INT8 to double the effective context length within the same VRAM budget.

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
  - **Task**: Add a "Context Provider" trait to the generation loop that injects retrieved snippets into the prompt dynamically.
- [ ] **Production Embedder**:
  - **Task**: Integration of a small BERT or similar model for high-quality embeddings (replacing the byte-hash MeanPoolEmbedder).

### 4. UI/UX Polish (`claude-tui`)
- [ ] **Markdown Rendering**:
  - **Task**: Use a crate like `pulldown-cmark` or `ratatui-markdown` to render formatted text in the chat window.
- [ ] **Horizontal Scrolling & Code Blocks**:
  - **Task**: Implement code block detection and syntax highlighting in the TUI using `syntect`.

---

## Execution Plan

### Immediate Next: Grouped Query Attention (GQA)
GQA reduces the number of KV heads, decreasing memory bandwidth requirements and enabling larger batch sizes for the same VRAM budget. This pairs naturally with the new long-context architecture.

**Plan**:
1. Add `n_kv_heads` field to `ModelConfig` (defaults to `n_head` for backward compatibility).
2. Modify `CausalSelfAttention` to use `n_kv_heads` for K/V projection and repeat-interleave for Q heads.
3. Update the `EvictingKVCache` to allocate based on `n_kv_heads` instead of `n_head`.
