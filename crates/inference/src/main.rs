use axum::{extract::State, routing::post, Json, Router};
use claude_core::ClaudeTransformer;
use inference::{load_model, Generator, SamplingParams};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tch::Device;
use tokenizer::BPE;

#[derive(Clone)]
struct AppState {
    model: Arc<ClaudeTransformer>,
    tokenizer: Arc<BPE>,
    device: Device,
}

#[derive(Deserialize)]
struct GenRequest {
    prompt: String,
    max_new_tokens: Option<usize>,
    max_input_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct GenResponse {
    text: String,
}

use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream};
use std::convert::Infallible;

async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut generator = Generator::new(Arc::clone(&state.model), state.device);
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(p) = req.top_p {
        params.top_p = p;
    }

    let max_tokens = req.max_new_tokens.unwrap_or(50);
    let max_input_tokens = req.max_input_tokens.unwrap_or(1024);

    let input_ids: Vec<i64> = state
        .tokenizer
        .encode_with_max_tokens(&req.prompt, max_input_tokens)
        .iter()
        .map(|&id| id as i64)
        .collect();

    if input_ids.is_empty() {
        let stream = stream::iter([Ok(Event::default().data(""))]);
        return Sse::new(stream);
    }

    let (tx, rx) = tokio::sync::mpsc::channel(max_tokens + 1);

    let input_ids_clone = input_ids.clone();

    tokio::task::spawn_blocking(move || {
        let _ = generator.generate_stream(&input_ids_clone, max_tokens, &params, tx);
    });

    let tokenizer = Arc::clone(&state.tokenizer);
    let stream = stream::unfold(rx, move |mut rx| {
        let tokenizer = Arc::clone(&tokenizer);
        async move {
            match rx.recv().await {
                Some(token_id) => {
                    let text = tokenizer.decode(&[token_id as u32]);
                    let event = Event::default().data(text);
                    Some((Ok(event), rx))
                }
                None => None,
            }
        }
    });

    Sse::new(stream)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let checkpoint_dir = std::path::Path::new("checkpoints");
    let vocab_path = "data/vocab.json";

    // 1. Load Tokenizer
    let tokenizer = if std::path::Path::new(vocab_path).exists() {
        println!("Loading tokenizer from {}", vocab_path);
        Arc::new(BPE::load(vocab_path)?)
    } else {
        println!("Warning: Tokenizer not found. Server may produce garbage.");
        Arc::new(BPE::new(
            tokenizer::Vocab::new(),
            std::collections::HashMap::new(),
        ))
    };

    // 2. Load Model
    let model = if checkpoint_dir.exists() && checkpoint_dir.join("config.json").exists() {
        Arc::new(load_model(checkpoint_dir, device)?)
    } else {
        println!("No model found. Initializing random one.");
        let config = claude_core::ModelConfig {
            n_embd: 128,
            n_head: 4,
            n_layer: 4,
            vocab_size: tokenizer.vocab.len() as i64,
            max_seq_len: 2048,
            dropout: 0.0,
            use_bias: true,
            layer_norm_epsilon: 1e-5,
        };
        let vs = tch::nn::VarStore::new(device);
        Arc::new(ClaudeTransformer::new(&vs.root(), &config))
    };

    let state = AppState {
        model,
        tokenizer,
        device,
    };

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    println!("Inference server listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
