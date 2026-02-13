# How to Contribute

We welcome contributions! Please follow these guidelines.

## Code Style

This project uses standard Rust formatting.
Run: `cargo fmt` before committing.

## Pull Requests

1.  **Fork** the repository.
2.  **Create a branch** (`feature/my-feature`).
3.  **Implement** your changes.
4.  **Add tests** in `crates/*/src/tests/` where applicable.
5.  **Run tests**: `cargo test --workspace`.
6.  **Create a Pull Request** targeting `main`.

## Design Philosophy

*   **Safety**: Prefer safe Rust unless `unsafe` is strictly necessary for perf (e.g., FFI).
*   **Performance**: Profile before optimizing. Use `criterion` for benchmarks.
*   **Modularity**: Keep crates decoupled. `claude-core` should not depend on `inference`.

## Roadmap

Planned features:
*   [ ] FlashAttention integration
*   [ ] Distributed Training (DDP)
*   [ ] Rust-native CUDA kernels (via `cudarc`)
*   [ ] WebGPU backend (via `burn-wgpu`)
