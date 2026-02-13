# Claude-Rust

A high-performance, terminal-integrated LLM inference and training engine built from scratch in Rust. This project implements a Transformer architecture with modern optimizations like RoPE, RMSNorm, and integrated KV caching.

## Tech Stack

![Rust](https://img.shields.io/badge/rust-%23E32F26.svg?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Tokio](https://img.shields.io/badge/Tokio-000000?style=for-the-badge&logo=tokio&logoColor=white)
![Axum](https://img.shields.io/badge/Axum-333333?style=for-the-badge)
<img width="945" height="926" alt="image" src="https://github.com/user-attachments/assets/4b4c561b-5507-4791-b33a-9a98ce4915eb" />

## Project Architecture

The system is organized as a workspace with specialized crates for modularity and performance:

- **claude-core**: The backbone of the system, containing the Transformer blocks, RoPE (Rotary Positional Embeddings), RMSNorm, and the in-place KV Cache logic.
- **inference**: A high-speed inference engine supporting SSE (Server-Sent Events) streaming via a web server and a flexible token generation pipeline.
- **tokenizer**: A custom Byte-Pair Encoding (BPE) implementation with advanced splitting rules and specialized training scripts.
- **trainer**: An autoregressive training crate with AdamW optimization and cross-entropy loss for fine-tuning.
- **retrieval**: A vector store implementation for Retrieval-Augmented Generation (RAG) using cosine similarity over tensor embeddings.
- **claude-tui**: A terminal-based user interface built with Ratatui for real-time interaction with the models.

## Key Features

- **In-place KV Caching**: Static allocation for key-value pair context to avoid O(N^2) memory reallocation overhead during long generation sessions.
- **Safetensors Support**: Integration with the Safetensors format for zero-copy, memory-mapped weight loading.
- **Streaming Inference**: SSE-based streaming server for character-by-character output in the TUI and other clients.
- **Custom BPE Logic**: Fully internal tokenizer logic, removing dependencies on external Python-based tokenizing tools.
- **RoPE Implementation**: Modern rotary positional embeddings for better long-context understanding.

## Getting Started

### Prerequisites

- Rust 1.70+
- LibTorch (Managed automatically via the tch-rs crate)

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

- **Quantization**: Implementation of INT8/4-bit linear quantization for model weights.
- **GQA (Grouped-Query Attention)**: Adding support for GQA to further reduce memory bandwidth usage.
- **Continuous Batching**: Refactoring the generator for high-throughput multi-request processing.
- **Markdown Rendering**: Integration of rich text formatting within the terminal interface.

## License

This project is licensed under the MIT License.

