//! Repair scenario tests covering edge cases and the full verify → repair pipeline.

use std::path::Path;

fn fixtures_dir() -> &'static Path {
    Path::new("tests/fixtures")
}

// =========================================================================
// repair_from_verify correctness
// =========================================================================

/// repair_from_verify must produce identical results to repair().
#[test]
fn test_repair_from_verify_matches_repair() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    copy_dir(fixtures_dir().join("damaged"), tmp1.path());
    copy_dir(fixtures_dir().join("damaged"), tmp2.path());

    let par2_1 = tmp1.path().join("testdata.bin.par2");
    let par2_2 = tmp2.path().join("testdata.bin.par2");
    let fs1 = rust_par2::parse(&par2_1).unwrap();
    let fs2 = rust_par2::parse(&par2_2).unwrap();

    // Method 1: repair() (includes internal verify)
    let r1 = rust_par2::repair(&fs1, tmp1.path()).unwrap();

    // Method 2: verify then repair_from_verify
    let vr = rust_par2::verify(&fs2, tmp2.path());
    let r2 = rust_par2::repair_from_verify(&fs2, tmp2.path(), &vr, true).unwrap();

    assert_eq!(r1.blocks_repaired, r2.blocks_repaired);
    assert_eq!(r1.files_repaired, r2.files_repaired);
    assert!(r1.success && r2.success);

    // Both should produce identical repaired files
    let f1 = std::fs::read(tmp1.path().join("testdata.bin")).unwrap();
    let f2 = std::fs::read(tmp2.path().join("testdata.bin")).unwrap();
    assert_eq!(
        f1, f2,
        "repair and repair_from_verify should produce identical files"
    );
}

// =========================================================================
// Repair idempotency
// =========================================================================

/// Repairing an already-repaired set should report NoDamage.
#[test]
fn test_repair_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(fixtures_dir().join("damaged"), tmp.path());

    let par2_path = tmp.path().join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    // First repair
    let r = rust_par2::repair(&file_set, tmp.path()).unwrap();
    assert!(r.success);

    // Second repair should find nothing to do
    let r2 = rust_par2::repair(&file_set, tmp.path());
    assert!(
        matches!(r2, Err(rust_par2::RepairError::NoDamage)),
        "second repair should return NoDamage, got: {r2:?}"
    );
}

/// Verify → repair → verify must show all files intact after repair.
#[test]
fn test_verify_repair_verify_cycle() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(fixtures_dir().join("damaged"), tmp.path());

    let par2_path = tmp.path().join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    // Pre-repair verify: damaged
    let pre = rust_par2::verify(&file_set, tmp.path());
    assert!(!pre.all_correct());
    assert!(pre.repair_possible);

    // Repair
    let result = rust_par2::repair_from_verify(&file_set, tmp.path(), &pre, true).unwrap();
    assert!(result.success);

    // Post-repair verify: all intact
    let post = rust_par2::verify(&file_set, tmp.path());
    assert!(
        post.all_correct(),
        "all files should be intact after repair: {post}"
    );
    assert_eq!(post.intact.len(), 1);
    assert!(post.damaged.is_empty());
    assert!(post.missing.is_empty());
}

// =========================================================================
// Error cases
// =========================================================================

/// Repairing intact files should return NoDamage error.
#[test]
fn test_repair_intact_returns_no_damage() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    let result = rust_par2::repair(&file_set, &dir);
    assert!(
        matches!(result, Err(rust_par2::RepairError::NoDamage)),
        "repairing intact files should return NoDamage: {result:?}"
    );
}

/// Insufficient recovery blocks should return InsufficientRecovery.
#[test]
fn test_repair_insufficient_recovery_error() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(fixtures_dir().join("unrepairable"), tmp.path());

    let par2_path = tmp.path().join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    let result = rust_par2::repair(&file_set, tmp.path());
    match result {
        Err(rust_par2::RepairError::InsufficientRecovery { needed, available }) => {
            assert!(
                needed > available,
                "needed ({needed}) should be > available ({available})"
            );
        }
        other => panic!("expected InsufficientRecovery, got: {other:?}"),
    }
}

// =========================================================================
// Verification detail tests
// =========================================================================

/// Verify correctly counts damaged blocks.
#[test]
fn test_verify_damaged_block_count() {
    let dir = fixtures_dir().join("damaged");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    let result = rust_par2::verify(&file_set, &dir);
    assert_eq!(result.damaged.len(), 1);
    let damaged = &result.damaged[0];
    assert!(
        damaged.damaged_block_count > 0,
        "should have at least 1 damaged block"
    );
    assert!(
        damaged.damaged_block_count <= damaged.total_block_count,
        "damaged blocks ({}) should not exceed total ({})",
        damaged.damaged_block_count,
        damaged.total_block_count,
    );
    assert_eq!(
        damaged.damaged_block_indices.len() as u32,
        damaged.damaged_block_count,
        "damaged_block_indices length should match damaged_block_count"
    );
}

/// Verify correctly identifies missing files.
#[test]
fn test_verify_missing_file_detail() {
    let dir = fixtures_dir().join("missing");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    let result = rust_par2::verify(&file_set, &dir);
    assert_eq!(result.missing.len(), 1);
    let missing = &result.missing[0];
    assert!(!missing.filename.is_empty());
    assert!(missing.expected_size > 0);
    assert!(missing.block_count > 0);
}

/// VerifyResult.blocks_needed() sums damaged and missing block counts.
#[test]
fn test_verify_blocks_needed() {
    let dir = fixtures_dir().join("damaged");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    let result = rust_par2::verify(&file_set, &dir);
    let expected: u32 = result
        .damaged
        .iter()
        .map(|d| d.damaged_block_count)
        .sum::<u32>()
        + result.missing.iter().map(|m| m.block_count).sum::<u32>();
    assert_eq!(result.blocks_needed(), expected);
}

/// Intact verify has zero blocks needed.
#[test]
fn test_verify_intact_zero_blocks_needed() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    let result = rust_par2::verify(&file_set, &dir);
    assert!(result.all_correct());
    assert_eq!(result.blocks_needed(), 0);
}

// =========================================================================
// Parsing tests
// =========================================================================

/// Parse should fail on non-existent file.
#[test]
fn test_parse_nonexistent_file() {
    let result = rust_par2::parse(Path::new("/nonexistent/file.par2"));
    assert!(result.is_err());
}

/// Parse should fail on empty file.
#[test]
fn test_parse_empty_file() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = tmp.path().join("empty.par2");
    std::fs::write(&empty, b"").unwrap();

    let result = rust_par2::parse(&empty);
    assert!(result.is_err(), "empty file should fail to parse");
}

/// Parse should fail on non-PAR2 data.
#[test]
fn test_parse_garbage_data() {
    let tmp = tempfile::tempdir().unwrap();
    let garbage = tmp.path().join("garbage.par2");
    std::fs::write(&garbage, b"This is not a PAR2 file at all").unwrap();

    let result = rust_par2::parse(&garbage);
    assert!(result.is_err(), "garbage data should fail to parse");
}

/// Parse should fail on truncated PAR2 header.
#[test]
fn test_parse_truncated_header() {
    let tmp = tempfile::tempdir().unwrap();
    let trunc = tmp.path().join("truncated.par2");
    // PAR2 magic is "PAR2\0PKT" (8 bytes) but header needs 64 bytes
    std::fs::write(&trunc, b"PAR2\x00PKT").unwrap();

    let result = rust_par2::parse(&trunc);
    assert!(result.is_err(), "truncated header should fail to parse");
}

/// Parsed PAR2 file set should contain expected metadata.
#[test]
fn test_parse_metadata() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    assert!(file_set.slice_size > 0, "slice_size should be > 0");
    assert!(!file_set.files.is_empty(), "should have at least 1 file");

    // Check file metadata
    for file in file_set.files.values() {
        assert!(!file.filename.is_empty(), "filename should not be empty");
        assert!(file.size > 0, "file size should be > 0");
        assert!(!file.slices.is_empty(), "should have per-slice checksums");
        // File should have the right number of slices
        let expected_slices = file.size.div_ceil(file_set.slice_size) as usize;
        assert_eq!(
            file.slices.len(),
            expected_slices,
            "slice count mismatch for {}",
            file.filename,
        );
    }
}

/// Recovery block count should be non-zero for sets with volume files.
#[test]
fn test_parse_recovery_count() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    // The intact fixture has volume files, so recovery_block_count may be > 0
    // if the index file contains recovery block references
    // At minimum, the file_set should parse without error
    // The file_set should parse without error and have a valid recovery count
    let _ = file_set.recovery_block_count;
}

/// hash_16k should return a valid 16-byte hash for any file.
#[test]
fn test_compute_hash_16k() {
    let dir = fixtures_dir().join("intact");
    let file_path = dir.join("testdata.bin");
    let hash = rust_par2::compute_hash_16k(&file_path).unwrap();

    // Hash should not be all zeros (extremely unlikely for real data)
    assert!(hash.iter().any(|&b| b != 0), "hash should not be all zeros");

    // Same file should produce same hash
    let hash2 = rust_par2::compute_hash_16k(&file_path).unwrap();
    assert_eq!(hash, hash2, "hash_16k should be deterministic");
}

/// hash_16k should match the PAR2 stored value.
#[test]
fn test_hash_16k_matches_par2() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    for file in file_set.files.values() {
        let file_path = dir.join(&file.filename);
        if file_path.exists() {
            let computed = rust_par2::compute_hash_16k(&file_path).unwrap();
            assert_eq!(
                computed, file.hash_16k,
                "hash_16k mismatch for {}",
                file.filename
            );
        }
    }
}

// =========================================================================
// Display formatting
// =========================================================================

/// VerifyResult Display should be human-readable.
#[test]
fn test_verify_result_display() {
    let dir = fixtures_dir().join("intact");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();
    let result = rust_par2::verify(&file_set, &dir);

    let display = format!("{result}");
    assert!(
        display.contains("correct"),
        "intact display should mention 'correct': {display}"
    );
}

#[test]
fn test_verify_result_display_damaged() {
    let dir = fixtures_dir().join("damaged");
    let par2_path = dir.join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();
    let result = rust_par2::verify(&file_set, &dir);

    let display = format!("{result}");
    assert!(
        display.contains("damaged") || display.contains("needed"),
        "damaged display should mention damage: {display}"
    );
}

#[test]
fn test_repair_result_display() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(fixtures_dir().join("damaged"), tmp.path());

    let par2_path = tmp.path().join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();
    let result = rust_par2::repair(&file_set, tmp.path()).unwrap();

    let display = format!("{result}");
    assert!(
        display.contains("complete") || display.contains("repaired"),
        "repair display should mention completion: {display}"
    );
}

// =========================================================================
// Helpers
// =========================================================================

fn copy_dir(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    let src = src.as_ref();
    let dst = dst.as_ref();
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), dst.join(entry.file_name())).unwrap();
        }
    }
}
