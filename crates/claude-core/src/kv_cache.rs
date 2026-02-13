use tch::{Tensor, Device, Kind};

pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub length: usize,
    pub max_capacity: usize,
}

impl KVCache {
    pub fn new(max_capacity: usize, n_head: i64, head_dim: i64, device: Device, kind: Kind) -> Self {
        let k = Tensor::zeros(&[1, n_head, max_capacity as i64, head_dim], (kind, device));
        let v = Tensor::zeros(&[1, n_head, max_capacity as i64, head_dim], (kind, device));
        Self {
            k,
            v,
            length: 0,
            max_capacity,
        }
    }

    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) {
        let _batch_size = new_k.size()[0];
        let _n_head = new_k.size()[1];
        let seq_len = new_k.size()[2];
        let _head_dim = new_k.size()[3];

        // If batch size changes or we exceed capacity, we might need a more complex strategy
        // For now, assume batch size 1 and increment length
        let start = self.length as i64;
        let end = start + seq_len;

        if end > self.max_capacity as i64 {
            // Simple truncation for now (FIFO-ish) - in reality we'd error or rotate
            return;
        }

        let _ = self.k.narrow(2, start, seq_len).copy_(new_k);
        let _ = self.v.narrow(2, start, seq_len).copy_(new_v);
        
        self.length += seq_len as usize;
    }

    pub fn get_view(&self) -> (Tensor, Tensor) {
        let k = self.k.narrow(2, 0, self.length as i64);
        let v = self.v.narrow(2, 0, self.length as i64);
        (k, v)
    }
    
    pub fn clear(&mut self) {
        self.length = 0;
    }
}
