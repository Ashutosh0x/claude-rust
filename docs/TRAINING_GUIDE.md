# Training Guide: Claude-Rust

This guide details the complete process for training a Claude-style LLM from scratch using Pure Rust.

## Prerequisites

1.  **Hardware**:
    *   **GPU**: NVIDIA GPU (recommended) with CUDA drivers installed (for `tch` backend).
    *   **CPU**: High-performance multi-core CPU (recommended for tokenization/preprocessing).
    *   **RAM**: Enough to memory-map your dataset (typically 16-64GB+).
    *   **Disk**: Fast SSD for training data (to avoid I/O bottlenecks).

2.  **Software**:
    *   Rust Toolchain (`rustup update`).
    *   (Optional) Python/Jupyter for analysis (notebooks).

## Step 1: Data Preparation

Data is king. You need a large, clean text corpus.

1.  **Collect Raw Data**:
    *   Place `.txt` files in `data/raw/`.
    *   Example: `data/raw/wiki_en.txt`, `data/raw/books.txt`.
    *   Format: Plain UTF-8 text. One document per file or concatenated.

2.  **Train Tokenizer**:
    *   Learn subword merges from your specialized corpus.
    *   **Command**:
        ```bash
        cargo run --release --bin tokenizer_cli train \
            --files "data/raw/*.txt" \
            --output-dir "data/processed" \
            --vocab-size 50257  # GPT-2 size
        ```
    *   **Output**: `data/processed/vocab.json`, `data/processed/merges.txt`.

3.  **Tokenize & Binarize**:
    *   Convert text to efficient `.bin` format (u32 array).
    *   **Command**:
        ```bash
        cargo run --release --bin data_prep -- \
            --input "data/raw/*.txt" \
            --output "data/processed/train.bin" \
            --val-output "data/processed/val.bin" \
            --val-split 0.05 \
            --vocab "data/processed/vocab.json" \
            --merges "data/processed/merges.txt"
        ```
    *   **Output**: `train.bin` (95%), `val.bin` (5%).

## Step 2: Configuration

Edit `configs/training_config.yaml` to define model hyperparameters.

```yaml
model_name: "claude-rust-small"

# Data
data_path: "data/processed/train.bin"
val_data_path: "data/processed/val.bin"

# Architecture
vocab_size: 50257
context_window: 1024
n_embd: 768       # Embedding dimension (GPT-2 Small: 768, Medium: 1024)
n_head: 12        # Number of attention heads
n_layer: 12       # Number of transformer blocks
dropout: 0.1

# Training
batch_size: 32    # Adjust based on VRAM
learning_rate: 6e-4
min_lr: 6e-5      # Cosine decay minimum
max_iters: 50000  # Total training steps
warmup_iters: 1000
weight_decay: 0.1
grad_clip: 1.0

# Checkpointing
out_dir: "checkpoints"
eval_interval: 500
log_interval: 10
always_save_checkpoint: true
save_top_k: 3     # Keep only best 3 checkpoints
```

## Step 3: Launch Training

Start the training loop:

```bash
cargo run --release --bin trainer -- --config configs/training_config.yaml
```

**Monitoring**:
*   The trainer outputs logs to stdout:
    ```
    [INFO] step 0 | loss: 10.452 | lr: 0.000000 | time: 245ms
    [INFO] step 10 | loss: 8.231 | lr: 0.000006 | time: 231ms
    ...
    [INFO] Saving checkpoint to checkpoints/ckpt_step_500.safetensors
    ```
*   Use `tail -f training.log` (if redirecting output) or parse JSON lines for plotting.

## Step 4: Resume Training

If interrupted, you can resume from the latest checkpoint:

```bash
cargo run --release --bin trainer -- \
    --config configs/training_config.yaml \
    --resume-from checkpoints/ckpt_latest.safetensors
```

## Step 5: Evaluation & Testing

Periodically, the trainer evaluates the model on the validation set (`val.bin`).
Lower `val_loss` indicates better generalization.

**Manual Testing**:
Once trained, verify generation quality:

```bash
cargo run --release --bin inference -- \
    --model checkpoints/ckpt_best.safetensors \
    --prompt "Once upon a time in a rust compiler," \
    --max-new-tokens 100
```
