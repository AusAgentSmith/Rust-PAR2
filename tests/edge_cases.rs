//! Edge case tests for PAR2 parsing and verification.

use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

fn fixtures_dir() -> &'static Path {
    Path::new("tests/fixtures")
}

#[test]
fn test_parse_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.par2");
    std::fs::File::create(&path).unwrap();

    let result = rust_par2::parse(&path);
    assert!(result.is_err(), "empty file should fail to parse");
}

#[test]
fn test_parse_truncated_par2() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("truncated.par2");
    let mut f = std::fs::File::create(&path).unwrap();
    // Write a partial PAR2 magic header (real magic is "PAR2\0PKT")
    f.write_all(b"PAR2\0PK").unwrap();
    drop(f);

    let result = rust_par2::parse(&path);
    assert!(result.is_err(), "truncated PAR2 should fail to parse");
}

#[test]
fn test_parse_nonexistent_file() {
    let result = rust_par2::parse(Path::new("/nonexistent/path/file.par2"));
    assert!(result.is_err(), "nonexistent file should fail");
}

#[test]
fn test_verify_intact_zero_blocks_needed() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");

    let file_set = rust_par2::parse(&par2_path).expect("Failed to parse");
    let result = rust_par2::verify(&file_set, &dir);

    assert!(result.all_correct());
    assert_eq!(
        result.blocks_needed(),
        0,
        "intact files should need 0 repair blocks"
    );
}
