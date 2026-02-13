# Deployment Guide: Claude-Rust Inference Server

This guide explains how to deploy the trained Claude-Rust model for production inference.

## Prerequisites

1.  **Docker**: Installed and running (Docker Desktop or Linux CLI).
2.  **GPU Drivers**: If deploying with NVIDIA GPU acceleration, ensure `nvidia-docker` (or NVIDIA Container Toolkit) is installed on the host.
    *   Test: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3.  **Trained Model**: A checkpoint file (`.safetensors`) and `vocab.json` ready in `checkpoints/`.

## Deployment Options

### 1. Docker Compose (Recommended for Local Dev/Small Scale)

This setups the inference API and (optionally) a frontend/database.

**`docker-compose.yml`**:
```yaml
version: '3.8'
services:
  inference:
    build: .
    image: claude-rust:latest
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - RUST_LOG=info
      - CHECKPOINT_PATH=/app/checkpoints/best_model.safetensors
      - CONFIG_PATH=/app/configs/model_config.yaml
    command: ["inference"]
```

**Running**:
```bash
docker-compose up --build -d
```
Check logs: `docker-compose logs -f inference`

### 2. Standalone Binary (Metal/Bare Metal)

For maximum performance on a dedicated server (AWS EC2 g4dn/p3, Lambda Labs, RunPod).

1.  **Build Release Binary**:
    ```bash
    cargo build --release --bin inference
    ```
    Binary location: `target/release/inference`

2.  **Run**:
    ```bash
    # Ensure CUDA libraries are in LD_LIBRARY_PATH if using tch
    export CHECKPOINT_PATH="./checkpoints/best_model.safetensors"
    ./target/release/inference --port 8080 --host 0.0.0.0
    ```

### 3. Kubernetes (K8s) (Scaling)

For high-traffic deployments, use K8s to manage replicas.

**Deployment Manifest (`k8s/deployment.yaml`)**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-rust-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-rust
  template:
    metadata:
      labels:
        app: claude-rust
    spec:
      containers:
      - name: inference
        image: your-registry/claude-rust:v1
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU per pod
        env:
        - name: CHECKPOINT_PATH
          value: "/models/v1.safetensors"
        volumeMounts:
        - name: model-volume
          mountPath: /models
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

## Performance Tuning

*   **Batch Size**: Increase `batch_size` in `server_config.yaml` to improve throughput at the cost of latency.
*   **Threads**: Set `OMP_NUM_THREADS` (OpenMP) to match physical CPU cores if using CPU inference.
*   **Quantization**:
    *   Use `crates/quant` tools to convert `fp32` weights to `int8` or `q4`.
    *   This reduces VRAM usage (e.g., 7B model fits on <6GB VRAM instead of 14GB).

## Security Best Practices

*   **Rate Limiting**: Add a reverse proxy (Nginx or Cloudflare) in front of the API to prevent abuse.
*   **Authentication**: Implement API keys middleware in `crates/inference/src/server.rs` or via API Gateway.
*   **Input Validation**: `max_new_tokens` is capped to prevent OOM/endless generation attacks (default: 2048).
