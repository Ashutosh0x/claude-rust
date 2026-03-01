//! Common tensor operations and helpers.

/// Compute the number of elements in a shape.
pub fn numel(shape: &[i64]) -> i64 {
    shape.iter().product()
}

/// Compute the stride for a contiguous tensor of the given shape.
pub fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Check if two shapes are broadcastable.
pub fn broadcastable(a: &[i64], b: &[i64]) -> bool {
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[1]), 1);
    }

    #[test]
    fn test_strides() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_broadcastable() {
        assert!(broadcastable(&[1, 3], &[2, 3]));
        assert!(!broadcastable(&[2, 3], &[4, 3]));
    }
}
