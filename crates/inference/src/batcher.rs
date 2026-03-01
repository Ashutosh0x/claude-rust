use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration, Instant};

use crate::generator::Generator;
use claude_core::ClaudeTransformer;
use tch::Device;

pub struct Request {
    pub input_ids: Vec<i64>,
    pub max_tokens: usize,
    pub resp: oneshot::Sender<Vec<i64>>, // Returns the generated token sequence
}

pub struct Batcher {
    rx: mpsc::Receiver<Request>,
    max_batch: usize,
    max_latency: Duration,
    model: Arc<ClaudeTransformer>,
    device: Device,
}

impl Batcher {
    pub fn new(
        rx: mpsc::Receiver<Request>,
        max_batch: usize,
        max_latency_ms: u64,
        model: Arc<ClaudeTransformer>,
        device: Device,
    ) -> Self {
        Self {
            rx,
            max_batch,
            max_latency: Duration::from_millis(max_latency_ms),
            model,
            device,
        }
    }

    pub async fn run(mut self) {
        loop {
            // Block until at least one request arrives
            let first = match self.rx.recv().await {
                Some(r) => r,
                None => break, // Channel closed
            };

            let mut batch = vec![first];
            let start = Instant::now();

            // Collect more requests until batch is full or timeout hit
            while batch.len() < self.max_batch && start.elapsed() < self.max_latency {
                match timeout(self.max_latency - start.elapsed(), self.rx.recv()).await {
                    Ok(Some(r)) => batch.push(r),
                    _ => break, // Timeout hit or channel closed
                }
            }

            // Group processing
            let model = Arc::clone(&self.model);
            let device = self.device;
            
            // For continuous batching in a real scenario we'd do per-token yielding.
            // For now, we delegate the batched inputs to Generator::generate_batch
            // which processes the full matrix and returns results.
            
            tokio::task::spawn_blocking(move || {
                let mut generator = Generator::new(model, device);
                
                // Execute the batched generation
                let outputs = generator.generate_batch(&batch);
                
                // Send back results
                for (req, out) in batch.into_iter().zip(outputs.into_iter()) {
                    let _ = req.resp.send(out);
                }
            });
        }
    }
}
