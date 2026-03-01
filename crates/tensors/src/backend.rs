//! Backend trait for tensor operations.

use super::TensorDevice;

/// Trait that all tensor backends must implement.
pub trait TensorBackend {
    type Tensor;

    /// Create a zeros tensor of the given shape.
    fn zeros(shape: &[i64], device: TensorDevice) -> Self::Tensor;

    /// Create a ones tensor of the given shape.
    fn ones(shape: &[i64], device: TensorDevice) -> Self::Tensor;

    /// Matrix multiply.
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    /// Softmax along a dimension.
    fn softmax(x: &Self::Tensor, dim: i64) -> Self::Tensor;

    /// Layer normalization.
    fn layer_norm(x: &Self::Tensor, normalized_shape: &[i64], eps: f64) -> Self::Tensor;

    /// Element-wise addition.
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
}
