use tch::{Tensor, Device, Kind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

pub struct VectorStore {
    documents: Vec<Document>,
    embeddings: Option<Tensor>,
    device: Device,
}

impl VectorStore {
    pub fn new(device: Device) -> Self {
        Self {
            documents: Vec::new(),
            embeddings: None,
            device,
        }
    }

    pub fn add_documents(&mut self, docs: Vec<Document>, embeddings: Tensor) {
        self.documents.extend(docs);
        match &mut self.embeddings {
            Some(existing) => {
                let new_embeddings = embeddings.to(self.device);
                *existing = Tensor::cat(&[existing.shallow_clone(), new_embeddings], 0);
            }
            None => {
                self.embeddings = Some(embeddings.to(self.device));
            }
        }
    }

    /// Search for most similar documents using cosine similarity
    /// query_embedding: [dim] or [1, dim] tensor
    pub fn search(&self, query_embedding: &Tensor, top_k: usize) -> Vec<(&Document, f64)> {
        let embeddings = match &self.embeddings {
            Some(e) => e,
            None => return Vec::new(),
        };

        let q = query_embedding.to_device(self.device).view([1, -1]);
        
        // Normalize for cosine similarity
        let q_norm = q.pow_tensor_scalar(2.0).sum_dim_intlist(Some(&[-1][..]), false, Kind::Double).sqrt();
        let e_norm = embeddings.pow_tensor_scalar(2.0).sum_dim_intlist(Some(&[-1][..]), true, Kind::Double).sqrt();
        
        let q_unit = &q / (q_norm + 1e-8);
        let e_unit = embeddings / (e_norm + 1e-8);
        
        let scores = q_unit.matmul(&e_unit.transpose(0, 1)).view([-1]);
        let k = std::cmp::min(top_k, self.documents.len());
        
        let (top_scores, top_indices) = scores.topk(k as i64, 0, true, true);
        
        let scores_vec: Vec<f32> = Vec::<f32>::try_from(&top_scores).unwrap_or_default();
        let indices_vec: Vec<i64> = Vec::<i64>::try_from(&top_indices).unwrap_or_default();
        
        indices_vec.iter().zip(scores_vec.iter())
            .map(|(&idx, &score)| (&self.documents[idx as usize], score as f64))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }
}
