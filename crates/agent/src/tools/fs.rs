use anyhow::Result;
use serde_json::Value;
use std::fmt::Debug;

use async_trait::async_trait;
use crate::tools::Tool;

#[derive(Debug)]
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read the contents of a file. Args: {\"path\": \"...\"}"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = tokio::fs::read_to_string(path).await?;
        Ok(content)
    }
}

#[derive(Debug)]
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write content to a file. Args: {\"path\": \"...\", \"content\": \"...\"}"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
        tokio::fs::write(path, content).await?;
        Ok(format!("Successfully wrote to {}", path))
    }
}

#[derive(Debug)]
pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &'static str {
        "list_dir"
    }

    fn description(&self) -> &'static str {
        "List contents of a directory. Args: {\"path\": \"...\"}"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let mut entries = tokio::fs::read_dir(path).await?;
        let mut result = String::new();
        while let Some(entry) = entries.next_entry().await? {
            result.push_str(&format!("{}\n", entry.file_name().to_string_lossy()));
        }
        Ok(result)
    }
}

#[derive(Debug)]
pub struct SearchCodebaseTool;

#[async_trait]
impl Tool for SearchCodebaseTool {
    fn name(&self) -> &'static str {
        "search_codebase"
    }

    fn description(&self) -> &'static str {
        "Search for a pattern in the codebase. Args: {\"query\": \"...\"}"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        // Simple mock implementation: recursive search (limited)
        let mut result = String::new();
        result.push_str(&format!("--- Search Results for '{}' ---\n", query));
        
        // Just search the current directory for now
        let mut entries = tokio::fs::read_dir(".").await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    if content.contains(query) {
                        result.push_str(&format!("Match in file: {:?}\n", path));
                    }
                }
            }
        }
        
        if result.len() < 40 {
            result.push_str("No matches found in root directory.");
        }
        
        Ok(result)
    }
}
