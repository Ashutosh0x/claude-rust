use std::sync::Arc;
use anyhow::Result;
use serde_json::Value;
use tokio::sync::mpsc;

use claude_core::ClaudeTransformer;
use inference::{Generator, SamplingParams};
use tokenizer::BPE;
use tch::Device;

use crate::tools::Tool;

#[derive(Debug)]
pub enum AgentEvent {
    Token(String),
    ToolStart(String, String), // name, arguments
    ToolResult(String),
    ToolError(String),
    Finished,
}

pub struct Agent {
    tools: Vec<Box<dyn Tool>>,
    model: Arc<ClaudeTransformer>,
    tokenizer: Arc<BPE>,
    device: Device,
}

impl Agent {
    pub fn new(
        tools: Vec<Box<dyn Tool>>,
        model: Arc<ClaudeTransformer>,
        tokenizer: Arc<BPE>,
        device: Device,
    ) -> Self {
        Self {
            tools,
            model,
            tokenizer,
            device,
        }
    }

    pub async fn run_agentic_loop(
        &self,
        prompt: &str,
        max_tokens: usize,
        tx: mpsc::Sender<AgentEvent>,
    ) -> Result<()> {
        let mut context_ids: Vec<i64> = self.tokenizer.encode(prompt).iter().map(|&id| id as i64).collect();
        let params = SamplingParams::default();
        let mut total_tokens_generated = 0;

        loop {
            if total_tokens_generated >= max_tokens {
                break;
            }

            let (token_tx, mut token_rx) = mpsc::channel(100);
            
            let mut generator = Generator::new(Arc::clone(&self.model), self.device);
            let ctx = context_ids.clone();
            let limit = max_tokens - total_tokens_generated;
            let params_clone = params.clone();
            
            // Start generation
            tokio::task::spawn_blocking(move || {
                let _ = generator.generate_stream(&ctx, limit, &params_clone, token_tx);
            });

            let mut in_tool_call = false;
            let mut tool_buffer = String::new();
            let mut chunk_buffer = String::new();
            let mut invoked_tool = false;

            while let Some(token_id) = token_rx.recv().await {
                total_tokens_generated += 1;
                context_ids.push(token_id);

                let text = self.tokenizer.decode(&[token_id as u32]);
                chunk_buffer.push_str(&text);
                
                if !in_tool_call {
                    if chunk_buffer.contains("<tool_call>") {
                        in_tool_call = true;
                        
                        // Extract anything after <tool_call>
                        if let Some(idx) = chunk_buffer.find("<tool_call>") {
                            tool_buffer.push_str(&chunk_buffer[idx + "<tool_call>".len()..]);
                        }
                    } else if chunk_buffer.len() > 20 {
                        // send what we know is safe text (so we don't hold back the stream forever)
                        // This logic could be improved with a sliding window
                        let _ = tx.send(AgentEvent::Token(chunk_buffer.clone())).await;
                        chunk_buffer.clear();
                    }
                } else {
                    tool_buffer.push_str(&text);
                    if tool_buffer.contains("</tool_call>") {
                        invoked_tool = true;
                        break; // Step out of the reading loop to execute tool
                    }
                }
            }

            // Flush remaining normal buffer if not in tool call
            if !in_tool_call && !chunk_buffer.is_empty() {
                let _ = tx.send(AgentEvent::Token(chunk_buffer.clone())).await;
            }

            if !invoked_tool {
                // Natural stop (e.g., EOF or generated all max_tokens)
                break;
            }

            // Execute the Tool
            let json_str = tool_buffer.split("</tool_call>").next().unwrap_or("").trim();
            let result_text = match serde_json::from_str::<Value>(json_str) {
                Ok(parsed) => {
                    let name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let args = parsed.get("arguments").cloned().unwrap_or(Value::Null);

                    let _ = tx.send(AgentEvent::ToolStart(name.to_string(), args.to_string())).await;

                    if let Some(tool) = self.tools.iter().find(|t| t.name() == name) {
                        match tool.execute(args).await {
                            Ok(res) => {
                                let _ = tx.send(AgentEvent::ToolResult(res.clone())).await;
                                res
                            }
                            Err(e) => {
                                let err_msg = format!("Tool {} failed: {}", name, e);
                                let _ = tx.send(AgentEvent::ToolError(err_msg.clone())).await;
                                err_msg
                            }
                        }
                    } else {
                        let err_msg = format!("Tool not found: {}", name);
                        let _ = tx.send(AgentEvent::ToolError(err_msg.clone())).await;
                        err_msg
                    }
                }
                Err(e) => {
                    let err_msg = format!("Failed to parse tool call JSON: {}\nInput was: {}", e, json_str);
                    let _ = tx.send(AgentEvent::ToolError(err_msg.clone())).await;
                    err_msg
                }
            };

            // Format result and inject into context
            let result_injection = format!("\n<tool_result>\n{}\n</tool_result>\n", result_text);
            let encoded_result = self.tokenizer.encode(&result_injection);
            
            for &id in &encoded_result {
                context_ids.push(id as i64);
            }
            
            // Loop restarts and continues decoding based on the updated context_ids
        }

        let _ = tx.send(AgentEvent::Finished).await;
        Ok(())
    }
}
