use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::RwLock;

use crate::error::Result;
use crate::vocab::Vocab;

#[derive(Serialize, Deserialize)]
pub struct BPE {
    pub vocab: Vocab,
    #[serde(with = "merges_serde")]
    pub merges: HashMap<(String, String), u32>,
    #[serde(skip)]
    #[serde(default = "default_cache")]
    pub cache: RwLock<HashMap<String, Vec<String>>>, // Thread-safe cache
    #[serde(skip)]
    #[serde(default = "default_regex")]
    pub regex: Regex,
}

mod merges_serde {
    use super::*;
    use serde::{Serializer, Deserializer, Serialize, Deserialize};

    pub fn serialize<S>(merges: &HashMap<(String, String), u32>, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let as_vec: Vec<(&(String, String), &u32)> = merges.iter().collect();
        as_vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<HashMap<(String, String), u32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let as_vec: Vec<((String, String), u32)> = Vec::deserialize(deserializer)?;
        Ok(as_vec.into_iter().collect())
    }
}

fn default_cache() -> RwLock<HashMap<String, Vec<String>>> {
    RwLock::new(HashMap::new())
}

impl Clone for BPE {
    fn clone(&self) -> Self {
        let cache_snapshot = self
            .cache
            .read()
            .map(|cache| cache.clone())
            .unwrap_or_default();

        Self {
            vocab: self.vocab.clone(),
            merges: self.merges.clone(),
            cache: RwLock::new(cache_snapshot),
            regex: self.regex.clone(),
        }
    }
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
            cache: default_cache(),
            regex: default_regex(),
        }
    }

    pub fn from_files<P: AsRef<Path>>(vocab_path: P, merges_path: P) -> Result<Self> {
        let vocab = Vocab::load(vocab_path)?;

        let file = File::open(merges_path)?;
        let reader = BufReader::new(file);
        let mut merges = HashMap::new();
        let mut merge_rank = 0u32;

        for line_res in reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.is_empty() {
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() == 2 {
                merges.insert((parts[0].to_string(), parts[1].to_string()), merge_rank);
                merge_rank += 1;
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
        if let Ok(cache) = self.cache.read() {
            if let Some(cached) = cache.get(token) {
                return cached.clone();
            }
        }

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
                if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
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

        if let Ok(mut cache) = self.cache.write() {
            cache.insert(token.to_string(), word.clone());
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

    pub fn encode_with_max_tokens(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        let mut ids = self.encode(text);
        if ids.len() > max_tokens {
            ids.truncate(max_tokens);
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
        bpe.cache = default_cache();
        bpe.regex = default_regex();
        Ok(bpe)
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn encode_with_max_tokens_respects_limit() {
        let mut vocab = Vocab::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);

        let bpe = BPE::new(vocab, HashMap::new());
        let ids = bpe.encode_with_max_tokens("ababa", 3);

        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn encode_populates_internal_cache() {
        let mut vocab = Vocab::new();
        vocab.insert("a".to_string(), 0);

        let bpe = BPE::new(vocab, HashMap::new());
        let _ = bpe.encode("a a");

        let cache = bpe.cache.read().expect("cache read lock");
        assert!(cache.contains_key("a"));
    }

    #[test]
    fn from_files_assigns_contiguous_merge_ranks_ignoring_comments_and_blanks() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("tokenizer_bpe_test_{unique}"));
        fs::create_dir_all(&dir).expect("create temp test dir");

        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");

        fs::write(&vocab_path, r#"{"a":0,"b":1,"ab":2}"#).expect("write vocab");
        fs::write(&merges_path, "#version: 0.2\n\n# comment\na b\n\n").expect("write merges");

        let bpe = BPE::from_files(&vocab_path, &merges_path).expect("load bpe from files");
        let rank = bpe.merges.get(&("a".to_string(), "b".to_string())).copied();

        assert_eq!(rank, Some(0));

        fs::remove_dir_all(&dir).expect("cleanup temp test dir");
    }
}
