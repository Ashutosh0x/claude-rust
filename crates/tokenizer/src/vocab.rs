use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
}

impl Vocab {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
        }
    }

    pub fn insert(&mut self, token: String, id: u32) {
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
    }

    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u32) -> Option<&String> {
        self.id_to_token.get(&id)
    }

    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.token_to_id)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let token_to_id: HashMap<String, u32> = serde_json::from_reader(reader)?;
        
        let mut id_to_token = HashMap::new();
        for (token, id) in &token_to_id {
            id_to_token.insert(*id, token.clone());
        }

        Ok(Self {
            token_to_id,
            id_to_token,
        })
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}
