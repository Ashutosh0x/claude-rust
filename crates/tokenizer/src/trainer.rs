use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::bpe::BPE;
use crate::error::Result;
use crate::vocab::Vocab;

pub struct Trainer {
    vocab_size: usize,
    min_frequency: u32,
    special_tokens: Vec<String>,
}

impl Trainer {
    pub fn new(vocab_size: usize, min_frequency: u32, special_tokens: Vec<String>) -> Self {
        Self {
            vocab_size,
            min_frequency,
            special_tokens,
        }
    }

    pub fn train(&self, files: &[String]) -> Result<BPE> {
        let regex = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")?;
        
        // 1. Read files and count words
        println!("Reading files and counting words...");
        let mut word_counts: HashMap<String, u32> = HashMap::new();
        
        for path in files {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                for mat in regex.find_iter(&line) {
                    let word = mat.as_str().to_string();
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }
        println!("Unique words: {}", word_counts.len());

        // 2. Initial split of words into chars
        let mut split_words: HashMap<String, Vec<String>> = HashMap::new();
        for (word, _) in &word_counts {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            split_words.insert(word.clone(), chars);
        }

        // 3. Initialize vocab with characters and special tokens
        let mut vocab = Vocab::new();
        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        
        // Add special tokens first
        for (i, token) in self.special_tokens.iter().enumerate() {
            vocab.insert(token.clone(), i as u32);
        }
        
        // Add base characters from corpus to vocab
        let mut base_chars: HashSet<String> = HashSet::new();
        for words in split_words.values() {
            for char_s in words {
                base_chars.insert(char_s.clone());
            }
        }
        
        for char_s in base_chars {
            if vocab.get_id(&char_s).is_none() {
                vocab.insert(char_s, vocab.len() as u32);
            }
        }

        // Add byte fallback tokens just in case (<0x00> to <0xFF>)
        for i in 0..256 {
            let s = format!("<0x{:02X}>", i);
            if vocab.get_id(&s).is_none() {
                vocab.insert(s, vocab.len() as u32);
            }
        }

        println!("Initial vocab size: {}", vocab.len());

        // 4. BPE Training Loop
        let mut current_vocab_size = vocab.len();
        let mut merge_count = 0;
        
        while current_vocab_size < self.vocab_size {
            // Count pairs
            let mut pair_counts: HashMap<(String, String), u32> = HashMap::new();
            
            for (word, count) in &word_counts {
                if let Some(tokens) = split_words.get(word) {
                    if tokens.len() < 2 {
                        continue;
                    }
                    for i in 0..tokens.len() - 1 {
                        let pair = (tokens[i].clone(), tokens[i + 1].clone());
                        *pair_counts.entry(pair).or_insert(0) += count;
                    }
                }
            }

            // Find best pair
            let mut best_pair: Option<(String, String)> = None;
            let mut max_count = 0;
            
            for (pair, count) in &pair_counts {
                if *count > max_count && *count >= self.min_frequency {
                    max_count = *count;
                    best_pair = Some(pair.clone());
                }
            }

            if best_pair.is_none() {
                println!("No more pairs to merge. Stopping.");
                break;
            }

            let (first, second) = best_pair.unwrap();
            let new_token = format!("{}{}", first, second);
            
            // Add to vocab
            vocab.insert(new_token.clone(), current_vocab_size as u32);
            merges.insert((first.clone(), second.clone()), merge_count); 
            merge_count += 1;
            
            // println!("Merging ({}, {}) -> {} (freq: {})", first, second, new_token, max_count);

            // Update split_words
            let words_to_update: Vec<String> = split_words.keys().cloned().collect();
            
            for word in words_to_update {
                if let Some(tokens) = split_words.get_mut(&word) {
                    let mut new_tokens = Vec::new();
                    let mut i = 0;
                    while i < tokens.len() {
                        if i < tokens.len() - 1 && tokens[i] == first && tokens[i + 1] == second {
                            new_tokens.push(new_token.clone());
                            i += 2;
                        } else {
                            new_tokens.push(tokens[i].clone());
                            i += 1;
                        }
                    }
                    *tokens = new_tokens;
                }
            }

            current_vocab_size += 1;
            if current_vocab_size % 100 == 0 {
                println!("Vocab size: {}", current_vocab_size);
            }
        }

        Ok(BPE::new(vocab, merges))
    }
}
