use std::fs::OpenOptions;
use std::io::Write;
use serde::Serialize;

/// Training metrics logger — writes metrics to stdout and a JSONL file.
#[derive(Clone)]
pub struct MetricsLogger {
    log_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepMetrics {
    pub step: usize,
    pub epoch: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub perplexity: f64,
    pub tokens_per_sec: Option<f64>,
}

impl MetricsLogger {
    pub fn new(log_path: Option<String>) -> Self {
        Self { log_path }
    }

    /// Log a step's metrics to stdout and optionally to a JSONL file.
    pub fn log(&self, metrics: &StepMetrics) {
        let tps_str = metrics
            .tokens_per_sec
            .map(|t| format!(" | {:.0} tok/s", t))
            .unwrap_or_default();

        println!(
            "[Step {:>6}] Epoch {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e}{}",
            metrics.step,
            metrics.epoch,
            metrics.loss,
            metrics.perplexity,
            metrics.learning_rate,
            tps_str,
        );

        if let Some(ref path) = self.log_path {
            if let Ok(json) = serde_json::to_string(metrics) {
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(file, "{}", json);
                }
            }
        }
    }

    /// Log an epoch summary.
    pub fn log_epoch_summary(&self, epoch: usize, avg_loss: f64, steps: usize) {
        let ppl = avg_loss.exp();
        println!(
            "══════ Epoch {} Complete | Avg Loss: {:.4} | PPL: {:.2} | Steps: {} ══════",
            epoch, avg_loss, ppl, steps
        );
    }
}

impl StepMetrics {
    pub fn new(step: usize, epoch: usize, loss: f64, learning_rate: f64) -> Self {
        Self {
            step,
            epoch,
            loss,
            learning_rate,
            perplexity: loss.exp(),
            tokens_per_sec: None,
        }
    }

    pub fn with_throughput(mut self, tokens_per_sec: f64) -> Self {
        self.tokens_per_sec = Some(tokens_per_sec);
        self
    }
}
