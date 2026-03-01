//! FAISS compatibility layer (stub for future integration).
//!
//! When scaling beyond ~100K vectors, replace FlatIndex with FAISS bindings.

/// Placeholder for FAISS index integration.
/// Use the `faiss` crate when ready for production-scale vector search.
pub struct FaissConfig {
    /// Number of Voronoi cells (IVF).
    pub nlist: usize,
    /// Number of cells to probe at search time.
    pub nprobe: usize,
    /// Use PQ (Product Quantization) compression.
    pub use_pq: bool,
    /// Number of PQ sub-quantizers.
    pub pq_m: usize,
}

impl Default for FaissConfig {
    fn default() -> Self {
        Self {
            nlist: 100,
            nprobe: 10,
            use_pq: false,
            pq_m: 8,
        }
    }
}

impl FaissConfig {
    /// Generate the FAISS index factory string for this config.
    pub fn factory_string(&self, _dim: usize) -> String {
        if self.use_pq {
            format!("IVF{},PQ{}", self.nlist, self.pq_m)
        } else {
            format!("IVF{},Flat", self.nlist)
        }
    }
}
