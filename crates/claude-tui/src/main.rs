use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use std::{error::Error, io, time::Duration};
use tokio::sync::mpsc;
use std::sync::Arc;

// Local crate imports
use claude_core::{ClaudeTransformer, ModelConfig};
use inference::{Generator, SamplingParams};
use tokenizer::{BPE, Vocab};
use tch::{nn, Device};
use tui_input::backend::crossterm::EventHandler;

use agent::{Agent, AgentEvent, Tool};
use agent::tools::fs::{ReadFileTool, WriteFileTool, ListDirTool, SearchCodebaseTool};
use agent::tools::cmd::RunCommandTool;

mod app;
mod ui;

use app::{App, Message, Sender};

#[derive(Debug)]
enum Action {
    Tick,
    TokenGenerated(String),
    SystemMessage(String),
    GenerationFinished,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 0. Initialize Model & Tokenizer
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);
    
    let vocab_path = "data/vocab.json";
    let checkpoint_dir = std::path::Path::new("checkpoints");
    
    // Load Tokenizer
    let tokenizer = if std::path::Path::new(vocab_path).exists() {
        println!("Loading tokenizer from {}", vocab_path);
        Arc::new(BPE::load(vocab_path)?)
    } else {
        println!("Warning: Tokenizer vocab not found at {}. Using minimal fallback.", vocab_path);
        let mut vocab = Vocab::new();
        vocab.insert(" ".to_string(), 32);
        for i in 65..123 {
            let c = i as u8 as char;
            vocab.insert(c.to_string(), i as u32);
        }
        vocab.insert("<UNK>".to_string(), 0);
        Arc::new(BPE::new(vocab, std::collections::HashMap::new()))
    };

    // Load Model
    let model = if checkpoint_dir.exists() && checkpoint_dir.join("config.json").exists() {
        Arc::new(inference::load_model(checkpoint_dir, device)?)
    } else {
        println!("Warning: No trained model found in {:?}. Initializing random model.", checkpoint_dir);
        let config = ModelConfig {
            n_embd: 128,
            n_head: 4,
            n_layer: 4,
            vocab_size: tokenizer.vocab.len() as i64,
            max_seq_len: 512,
            dropout: 0.1,
            use_bias: true,
            layer_norm_epsilon: 1e-5,
        };
        let vs = nn::VarStore::new(device);
        Arc::new(ClaudeTransformer::new(&vs.root(), &config))
    };

    // 1. Setup terminal (raw mode, alternate screen)
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 2. Setup channels and app
    let (tx, mut rx) = mpsc::channel(32);
    let mut app = App::new();

    // 3. Event Loop
    let mut reader = EventStream::new();
    let tick_rate = Duration::from_millis(100);
    let tx_tick = tx.clone();

    // Tick task
    tokio::spawn(async move {
        loop {
            if tx_tick.send(Action::Tick).await.is_err() {
                break;
            }
            tokio::time::sleep(tick_rate).await;
        }
    });

    // Run loop
    let res = run_app(&mut terminal, &mut app, &mut reader, tx, &mut rx, model, tokenizer, device).await;

    // 4. Cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    reader: &mut EventStream,
    tx: mpsc::Sender<Action>,
    rx: &mut mpsc::Receiver<Action>,
    model: Arc<ClaudeTransformer>,
    tokenizer: Arc<BPE>,
    device: Device,
) -> io::Result<()> {
    loop {
        // Draw
        terminal.draw(|f| ui::draw(f, app))?;

        // Handle events
        tokio::select! {
            // Priority: Internal Actions (Ticks, Responses)
            Some(action) = rx.recv() => {
                match action {
                    Action::Tick => {}
                    Action::TokenGenerated(token_text) => {
                        app.append_token(&token_text);
                    }
                    Action::SystemMessage(msg) => {
                        app.append_token(&format!("\n\x1b[33m[System: {}]\x1b[0m\n", msg));
                    }
                    Action::GenerationFinished => {
                        app.is_loading = false;
                    }
                }
            }
            // User Input
            Some(Ok(event)) = reader.next() => {
                match event {
                    Event::Key(key) => {
                        if key.code == KeyCode::Char('c') && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                            return Ok(());
                        }

                        match key.code {
                            KeyCode::Enter => {
                                let text: String = app.input.value().into();
                                if !text.trim().is_empty() {
                                    app.messages.push(Message {
                                        sender: Sender::User,
                                        content: text.clone(),
                                    });
                                    app.input.reset();
                                    app.is_loading = true;
                                    
                                    let tx_action = tx.clone();
                                    let model = Arc::clone(&model);
                                    let tokenizer = Arc::clone(&tokenizer);
                                    let prompt = text.clone();
                                    
                                    tokio::spawn(async move {
                                        let mut tools: Vec<Box<dyn Tool>> = Vec::new();
                                        tools.push(Box::new(ReadFileTool));
                                        tools.push(Box::new(WriteFileTool));
                                        tools.push(Box::new(ListDirTool));
                                        tools.push(Box::new(SearchCodebaseTool));
                                        tools.push(Box::new(RunCommandTool));

                                        let agent = Agent::new(tools, Arc::clone(&model), Arc::clone(&tokenizer), device);
                                        
                                        // 2. Setup internal stream channel
                                        let (token_tx, mut token_rx) = mpsc::channel(100);
                                        
                                        // 3. Start generation in a blocking-safe way if necessary or just await
                                        let tx_action_clone = tx_action.clone();
                                        
                                        tokio::spawn(async move {
                                            let _ = agent.run_agentic_loop(&prompt, 1024, token_tx).await;
                                        });

                                        while let Some(event) = token_rx.recv().await {
                                            match event {
                                                AgentEvent::Token(text) => {
                                                    let _ = tx_action_clone.send(Action::TokenGenerated(text)).await;
                                                }
                                                AgentEvent::ToolStart(name, args) => {
                                                    let _ = tx_action_clone.send(Action::SystemMessage(format!("Executing tool {} with args {}", name, args))).await;
                                                }
                                                AgentEvent::ToolResult(res) => {
                                                    let preview = if res.len() > 100 { format!("{}...", &res[0..100]).replace('\n', " ") } else { res.clone().replace('\n', " ") };
                                                    let _ = tx_action_clone.send(Action::SystemMessage(format!("Tool result returning {} chars (preview: {})", res.len(), preview))).await;
                                                }
                                                AgentEvent::ToolError(err) => {
                                                    let _ = tx_action_clone.send(Action::SystemMessage(format!("Tool error: {}", err))).await;
                                                }
                                                AgentEvent::Finished => {
                                                    break;
                                                }
                                            }
                                        }
                                        
                                        let _ = tx_action.send(Action::GenerationFinished).await;
                                    });
                                }
                            }
                            KeyCode::Esc => {
                                app.input.reset();
                            }
                            _ => {
                                app.input.handle_event(&Event::Key(key));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
