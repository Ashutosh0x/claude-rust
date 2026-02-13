use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::Result;
use crate::vocab::Vocab;

#[derive(Clone, Serialize, Deserialize)]
pub struct BPE {
    pub vocab: Vocab,
    pub merges: HashMap<(String, String), u32>,
    #[serde(skip)]
    pub cache: HashMap<String, Vec<String>>, // Simple cache
    #[serde(skip)]
    #[serde(default = "default_regex")]
    pub regex: Regex,
}

fn default_regex() -> Regex {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap()
}

// Custom Debug impl to skip regex
impl std::fmt::Debug for BPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BPE")
         .field("vocab_size", &self.vocab.len())
         .field("merges_count", &self.merges.len())
         .finish()
    }
}

impl BPE {
    pub fn new(vocab: Vocab, merges: HashMap<(String, String), u32>) -> Self {
        Self {
            vocab,
            merges,
            cache: HashMap::new(),
            regex: default_regex(),
        }
    }

    pub fn from_files<P: AsRef<Path>>(vocab_path: P, merges_path: P) -> Result<Self> {
        let vocab = Vocab::load(vocab_path)?;
        
        let file = File::open(merges_path)?;
        let reader = BufReader::new(file);
        let mut merges = HashMap::new();
        
        for (i, line_res) in reader.lines().enumerate() {
            let line = line_res?;
            if line.starts_with("#") || line.trim().is_empty() {
                continue;
            }
            // Skip version line if present
            if i == 0 && line.starts_with("#version") {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                merges.insert((parts[0].to_string(), parts[1].to_string()), i as u32);
            }
        }

        Ok(Self::new(vocab, merges))
    }

    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        if word.len() < 2 {
            return pairs;
        }
        for i in 0..word.len() - 1 {
            pairs.insert((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        // If word is in cache, return it
        // (This implementation requires RefCell for interior mutability to cache, skipping for simplicity in this immutable method)

        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        
        loop {
            let pairs = Self::get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            let mut best_pair: Option<(String, String)> = None;
            let mut min_rank = u32::MAX;

            for pair in &pairs {
                if let Some(&rank) = self.merges.get(pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some(pair.clone());
                    }
                }
            }

            if best_pair.is_none() {
                break;
            }

            let (first, second) = best_pair.unwrap();
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if i < word.len() - 1 && word[i] == first && word[i+1] == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;
            if word.len() == 1 {
                break;
            }
        }

        word
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for mat in self.regex.find_iter(text) {
            let token_text = mat.as_str();
            let bpe_tokens = self.bpe(token_text);
            
            for token in bpe_tokens {
                if let Some(id) = self.vocab.get_id(&token) {
                    ids.push(id);
                } else {
                    // Fallback: encode as bytes
                    for byte in token.bytes() {
                        let s = format!("<0x{:02X}>", byte); 
                        if let Some(id) = self.vocab.get_id(&s) {
                            ids.push(id);
                        } else if let Some(id) = self.vocab.get_id("<UNK>") {
                             ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }
    
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for id in ids {
            if let Some(token) = self.vocab.get_token(*id) {
                text.push_str(token);
            }
        }
        text
    }
    
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut bpe: BPE = serde_json::from_reader(reader)?;
        bpe.regex = default_regex();
        Ok(bpe)
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}
