use anyhow::Result;
use serde_json::Value;
use std::fmt::Debug;
use async_trait::async_trait;

pub mod fs;
pub mod cmd;

#[async_trait]
pub trait Tool: Send + Sync + Debug {
    /// The name of the tool (e.g., "read_file")
    fn name(&self) -> &'static str;
    
    /// A description of what the tool does and what arguments it expects (for the prompt)
    fn description(&self) -> &'static str;
    
    /// Execute the tool with the provided arguments (usually parsed as JSON)
    async fn execute(&self, args: Value) -> Result<String>;
}
