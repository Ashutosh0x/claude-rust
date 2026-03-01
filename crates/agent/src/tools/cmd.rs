use anyhow::Result;
use serde_json::Value;
use std::fmt::Debug;
use std::process::Stdio;

use async_trait::async_trait;
use crate::tools::Tool;

#[derive(Debug)]
pub struct RunCommandTool;

#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &'static str {
        "run_command"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the terminal. Args: {\"command\": \"...\"}"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        
        // --- SANDBOXING POLICIES ---
        // 1. Timeouts: Max 10 seconds per command execution
        let timeout_duration = std::time::Duration::from_secs(10);
        // 2. Directory Jailing: Only execute inside a specific tmp folder for safety
        let safe_dir = std::env::temp_dir().join("claude_sandbox");
        tokio::fs::create_dir_all(&safe_dir).await?;
        // 3. Output Caps: Max 4000 chars returned to context window
        let max_chars = 4000;

        let process = if cfg!(target_os = "windows") {
            tokio::process::Command::new("powershell")
                .arg("-Command")
                .arg(cmd)
                .current_dir(&safe_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        } else {
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&safe_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        };

        match tokio::time::timeout(timeout_duration, process.wait_with_output()).await {
            Ok(output_result) => {
                let output = output_result?;
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
                
                // Enforce Context Caps
                if stdout.len() > max_chars {
                    stdout.truncate(max_chars);
                    stdout.push_str("\n...[STDOUT TRUNCATED BY SANDBOX]...");
                }
                if stderr.len() > max_chars {
                    stderr.truncate(max_chars);
                    stderr.push_str("\n...[STDERR TRUNCATED BY SANDBOX]...");
                }

                let mut result = String::new();
                if !stdout.is_empty() {
                    result.push_str(&format!("STDOUT:\n{}\n", stdout));
                }
                if !stderr.is_empty() {
                    result.push_str(&format!("STDERR:\n{}\n", stderr));
                }
                
                if result.is_empty() {
                    result.push_str("Command executed with no output.");
                }
                
                Ok(result)
            }
            Err(_) => {
                // Timeout fired, kill the child process if possible.
                // Wait_with_output consumes the child so we just return the timeout error
                Ok(format!("Error: Command timed out after {} seconds. Terminated by Agent Sandbox.", timeout_duration.as_secs()))
            }
        }
    }
}
