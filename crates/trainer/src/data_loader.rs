use tch::{Tensor, Device};

/// Memory-mapped streaming data loader for large binary token files.
///
/// Reads pre-tokenized `.bin` files (sequences of u32 token IDs) and
/// yields sequential batches without loading the entire file into memory.
pub struct DataLoader {
    data: Vec<i64>,
    context_length: usize,
    batch_size: usize,
    device: Device,
    cursor: usize,
}

impl DataLoader {
    /// Load a binary token file (sequence of little-endian u32 IDs).
    pub fn from_bin_file(
        path: &str,
        context_length: usize,
        batch_size: usize,
        device: Device,
    ) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let n_tokens = bytes.len() / 4;

        let data: Vec<i64> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                u32::from_le_bytes(arr) as i64
            })
            .collect();

        println!(
            "[DataLoader] Loaded {} tokens from {} ({:.1} MB)",
            n_tokens,
            path,
            bytes.len() as f64 / 1_048_576.0
        );

        Ok(Self {
            data,
            context_length,
            batch_size,
            device,
            cursor: 0,
        })
    }

    /// Load from an in-memory token vector.
    pub fn from_tokens(
        tokens: Vec<i64>,
        context_length: usize,
        batch_size: usize,
        device: Device,
    ) -> Self {
        Self {
            data: tokens,
            context_length,
            batch_size,
            device,
            cursor: 0,
        }
    }

    /// Total number of tokens in the dataset.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Number of complete batches available.
    pub fn num_batches(&self) -> usize {
        let tokens_per_batch = self.batch_size * (self.context_length + 1);
        self.data.len().saturating_sub(1) / tokens_per_batch
    }

    /// Get the next sequential batch. Wraps around at end of data.
    pub fn next_batch(&mut self) -> (Tensor, Tensor) {
        let mut inputs = Vec::with_capacity(self.batch_size * self.context_length);
        let mut targets = Vec::with_capacity(self.batch_size * self.context_length);

        for _ in 0..self.batch_size {
            if self.cursor + self.context_length + 1 > self.data.len() {
                self.cursor = 0;
            }

            let chunk = &self.data[self.cursor..self.cursor + self.context_length + 1];
            inputs.extend_from_slice(&chunk[..self.context_length]);
            targets.extend_from_slice(&chunk[1..self.context_length + 1]);

            self.cursor += self.context_length;
        }

        let input_tensor = Tensor::from_slice(&inputs)
            .view([self.batch_size as i64, self.context_length as i64])
            .to(self.device);

        let target_tensor = Tensor::from_slice(&targets)
            .view([self.batch_size as i64, self.context_length as i64])
            .to(self.device);

        (input_tensor, target_tensor)
    }

    /// Reset the cursor to the beginning.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
