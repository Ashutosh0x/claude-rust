# Claude-Rust

A high-performance, terminal-integrated LLM inference and training engine built from scratch in Rust. This project implements a Transformer architecture with modern optimizations including **NTK-Aware RoPE**, **Sliding Window Attention**, **Evicting KV Cache with Sink Tokens**, RMSNorm, and support for up to **1M token context**.

## Tech Stack

![Rust](https://img.shields.io/badge/rust-%23E32F26.svg?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Tokio](https://img.shields.io/badge/Tokio-000000?style=for-the-badge&logo=tokio&logoColor=white)
![Axum](https://img.shields.io/badge/Axum-333333?style=for-the-badge)
<img width="945" height="926" alt="image" src="https://github.com/user-attachments/assets/4b4c561b-5507-4791-b33a-9a98ce4915eb" />

## Project Architecture

The system is organized as a workspace with specialized crates for modularity and performance:

- **claude-core**: The backbone of the system — Transformer blocks, NTK-Aware RoPE (Rotary Positional Embeddings), RMSNorm, Sliding Window Attention, Evicting KV Cache with Sink Token pinning, embedding layers, weight initialization, and sinusoidal/rotary position encodings.
- **inference**: A high-speed inference engine supporting SSE (Server-Sent Events) streaming via a web server, flexible token generation pipeline with unbounded context support, and configurable server/CLI modes.
- **tokenizer**: A custom Byte-Pair Encoding (BPE) implementation with advanced splitting rules, byte-level fallback encoding, and specialized training scripts.
- **trainer**: Full training pipeline with AdamW optimization, cosine LR scheduling with warmup, checkpoint save/resume, JSONL metrics logging, and streaming binary data loader.
- **retrieval**: RAG pipeline with text chunking, mean-pool embeddings, flat cosine similarity index, and FAISS-compatible configuration.
- **quant**: INT8 and 4-bit model quantization with per-group scaling (~4–8x compression) for consumer GPU inference.
- **tensors**: Backend abstraction layer with tensor operation utilities, broadcast checks, and CUDA/VRAM estimation.
- **agent**: A highly asynchronous orchestrator layer giving local models access to file system reading/writing and terminal execution natively.
- **claude-tui**: A terminal-based user interface built with Ratatui for real-time interaction with the models.
- **utils**: Shared config (YAML/JSON) loading, filesystem helpers, and logging setup.

## Key Features

| Feature | Description | Crate |
|---------|-------------|-------|
| **NTK-Aware RoPE Scaling** | Extends context beyond training length (e.g., 4K → 1M) by scaling base frequency while keeping local features intact. | `claude-core` |
| **Sliding Window Attention** | O(N × W) attention instead of O(N²) — each token attends to a local window + global sink tokens. | `claude-core` |
| **Evicting KV Cache** | Ring-buffer KV cache with sink token pinning — bounded memory, graceful eviction of middle tokens. | `claude-core` |
| **Attention Sink Tokens** | First N tokens are pinned as global context anchors, stabilizing attention quality over long contexts. | `claude-core` |
| **Safetensors Support** | Integration with the Safetensors format for lightweight, memory-mapped weight loading. | `claude-core` |
| **Agentic Tool Calling** | Intercepts `<tool_call>` generation to seamlessly read/write files and list directories. | `agent` |
| **Terminal Execution** | Native sub-process execution allowing the model to run bash/powershell commands on the system. | `agent` |
| **Streaming Inference** | SSE combined with Ratatui for real-time, char-by-char output streaming. | `inference` / `claude-tui` |
| **Custom BPE Logic** | Fully internal tokenizer implementation without external Python dependencies. | `tokenizer` |
| **INT8/Q4 Quantization** | Symmetric INT8 and per-group 4-bit quantization for ~4–8x model compression. | `quant` |
| **Training Pipeline** | AdamW + cosine LR scheduler + checkpoint save/resume + JSONL metrics logging. | `trainer` |
| **RAG Pipeline** | Text chunking, embeddings, flat cosine similarity search for retrieval-augmented generation. | `retrieval` |

## Long-Context Architecture

The system achieves 1M token context through a 4-component hybrid architecture:

```mermaid
flowchart TB
    Input["Input Tokens"] --> Emb["Embedding"]
    Emb --> Block

    subgraph Block["Transformer Block × N"]
        direction TB
        LN1["RMSNorm"] --> Attn["Sliding Window Attention\n+ Sink Tokens"]
        Attn --> Res1["+ Residual"]
        Res1 --> LN2["RMSNorm"]
        LN2 --> MLP["MLP (GeLU)"]
        MLP --> Res2["+ Residual"]
    end

    ROPE["NTK-Aware RoPE\n(position scaling)"] -.-> Attn
    MASK["Sliding Window Mask\n(local + sink)"] -.-> Attn
    KVC["Evicting KV Cache\n(ring buffer + sink pinning)"] -.-> Attn

    Block --> LNF["Final RMSNorm"]
    LNF --> LMHead["LM Head"]
    LMHead --> Output["Logits"]
```

| Component | Problem Solved | Without It |
|-----------|---------------|------------|
| NTK RoPE Scaling | Model sees positions > training length | Garbage output past training context |
| Sliding Window Attention | O(N²) → O(N × W) compute | OOM at ~100K tokens |
| Evicting KV Cache | Memory bounded, constant VRAM | OOM at ~50K tokens |
| Sink Tokens | Attention quality stays stable | Model loses coherence over long contexts |

### Hardware Requirements (7B Model)

| Context Length | GPU VRAM | Speed |
|---------------|----------|-------|
| 32K | ~24 GB | Fast |
| 128K | ~28 GB | Moderate |
| 512K | ~32 GB | Slow |
| 1M | ~40–48 GB | Very slow, feasible |

## Getting Started

### Prerequisites

- Rust 1.70+
- LibTorch (Managed automatically via the tch-rs crate)

### Configuration

Model configuration supports both standard and long-context setups:

```yaml
# configs/model_config.yaml
n_embd: 768
n_head: 12
n_layer: 12
vocab_size: 50257
max_seq_len: 2048
original_max_seq_len: 2048   # Training context length
rope_base: 10000.0           # RoPE base frequency
window_size: 2048            # Sliding window size
sink_tokens: 4               # Global context anchors
kv_cache_capacity: 2048      # Max KV cache tokens
```

For 1M context, see [`configs/million_context.toml`](configs/million_context.toml).

### Training a Tokenizer

Use the provided PowerShell script to train the BPE tokenizer on a raw text corpus:

```powershell
./scripts/train_tokenizer.ps1
```

### Running the Inference Server

Start the Axum-based inference server:

```bash
cargo run -p inference --bin inference-server
```

### Starting the TUI

Launch the terminal chat interface:

```bash
cargo run -p claude-tui
```

## Production Roadmap

- [x] **Long-Context Architecture**: NTK RoPE, Sliding Window Attention, Evicting KV Cache, Sink Tokens
- [x] **Quantization**: INT8 and 4-bit per-group quantization with calibration stats.
- [x] **Training Pipeline**: Checkpoint save/resume, cosine LR scheduler, JSONL metrics, streaming data loader.
- [x] **RAG Pipeline**: Text chunking, embedding, flat cosine similarity index.
- [ ] **GQA (Grouped-Query Attention)**: Adding support for GQA to further reduce memory bandwidth usage.
- [ ] **Continuous Batching**: Refactoring the generator for high-throughput multi-request processing.
- [ ] **Markdown Rendering**: Integration of rich text formatting within the terminal interface.

## License

This project is licensed under the MIT License.
