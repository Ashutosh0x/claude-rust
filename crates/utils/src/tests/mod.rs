// Utils crate tests.

#[cfg(test)]
mod fs_tests {
    use crate::fs::human_size;

    #[test]
    fn test_human_size_formatting() {
        assert_eq!(human_size(512), "512.0 B");
        assert_eq!(human_size(2048), "2.0 KB");
    }
}
