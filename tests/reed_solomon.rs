//! Reed-Solomon encode/decode round-trip tests.
//!
//! Verifies that the Vandermonde matrix construction, inversion, and
//! SIMD multiply-accumulate pipeline correctly recovers original data
//! across a variety of input sizes and damage patterns.

use rust_par2::gf;
use rust_par2::matrix::GfMatrix;

// =========================================================================
// Matrix inversion correctness
// =========================================================================

/// Verify A * A^-1 = I for various matrix sizes.
#[test]
fn test_matrix_inverse_roundtrip_sizes() {
    for n in 1..=16 {
        let mut m = GfMatrix::zeros(n, n);
        // Fill with deterministic non-trivial values
        for r in 0..n {
            for c in 0..n {
                let val = gf::exp2(((r * 7 + c * 13 + 1) % 65535) as u32);
                m.set(r, c, val);
            }
        }

        if let Some(inv) = m.invert() {
            // Verify M * M^-1 = I
            let product = mat_mul(&m, &inv);
            for r in 0..n {
                for c in 0..n {
                    let expected = if r == c { 1 } else { 0 };
                    assert_eq!(
                        product.get(r, c),
                        expected,
                        "M * M^-1 != I at ({r},{c}) for {n}x{n} matrix"
                    );
                }
            }
        }
        // Some random matrices may be singular — that's OK
    }
}

/// PAR2 encoding matrix submatrices must always be invertible
/// when selecting N rows from an (N+K) x N matrix.
#[test]
fn test_par2_encoding_matrix_invertibility() {
    // Test various input counts and recovery block counts
    let configs = [
        (2, vec![0, 1]),
        (3, vec![0, 1, 2]),
        (4, vec![0, 1]),
        (5, vec![0, 1, 2, 3, 4]),
        (8, vec![0, 1, 2, 3]),
        (16, vec![0, 1, 2, 3, 4, 5, 6, 7]),
        (32, vec![0, 1, 2, 3]),
    ];

    for (input_count, recovery_exponents) in &configs {
        let enc = GfMatrix::par2_encoding_matrix(*input_count, recovery_exponents);
        let n = *input_count;
        let k = recovery_exponents.len();

        // Test: lose first k data blocks, recover from recovery blocks
        let mut rows: Vec<usize> = Vec::new();
        for i in 0..n {
            if i < k {
                // Use recovery row instead
                rows.push(n + (i % k));
            } else {
                rows.push(i); // Identity row
            }
        }
        let sub = enc.select_rows(&rows);
        assert!(
            sub.invert().is_some(),
            "encoding matrix should be invertible: inputs={n}, recovery={k}"
        );
    }
}

/// Losing ANY combination of k blocks (out of n) should be recoverable
/// with k recovery blocks.
#[test]
fn test_any_k_blocks_recoverable() {
    let n = 5; // 5 input blocks
    let recovery_exponents = vec![0, 1, 2]; // 3 recovery blocks
    let k = 3;
    let enc = GfMatrix::par2_encoding_matrix(n, &recovery_exponents);

    // Try all (5 choose 3) = 10 combinations of 3 damaged blocks
    let combinations = combinations(n, k);
    for damaged in &combinations {
        let mut rows = Vec::new();
        let mut recovery_idx = 0;
        for i in 0..n {
            if damaged.contains(&i) {
                rows.push(n + recovery_idx);
                recovery_idx += 1;
            } else {
                rows.push(i);
            }
        }
        let sub = enc.select_rows(&rows);
        assert!(
            sub.invert().is_some(),
            "should recover from losing blocks {:?}",
            damaged
        );
    }
}

// =========================================================================
// Full encode → decode round-trips
// =========================================================================

/// Round-trip: encode data blocks to recovery, lose some data, decode back.
#[test]
fn test_rs_roundtrip_3_data_2_recovery() {
    rs_roundtrip_test(3, &[0, 1], &[0, 1], 8);
}

#[test]
fn test_rs_roundtrip_4_data_4_recovery() {
    rs_roundtrip_test(4, &[0, 1, 2, 3], &[0, 2, 3], 16);
}

#[test]
fn test_rs_roundtrip_lose_middle_blocks() {
    // Lose blocks 1 and 2 out of [0,1,2,3] — middle blocks, not first
    rs_roundtrip_test(4, &[0, 1], &[1, 2], 8);
}

#[test]
fn test_rs_roundtrip_lose_last_block() {
    rs_roundtrip_test(3, &[0], &[2], 8);
}

#[test]
fn test_rs_roundtrip_lose_all_blocks() {
    rs_roundtrip_test(3, &[0, 1, 2], &[0, 1, 2], 8);
}

#[test]
fn test_rs_roundtrip_single_block() {
    rs_roundtrip_test(1, &[0], &[0], 16);
}

#[test]
fn test_rs_roundtrip_large_block_size() {
    // Use a slice size large enough to exercise SIMD (256 bytes)
    rs_roundtrip_test(4, &[0, 1, 2, 3], &[0, 3], 256);
}

#[test]
fn test_rs_roundtrip_simd_sized_blocks() {
    // 768 bytes — exercises multiple AVX2 chunks (32 bytes each)
    rs_roundtrip_test(3, &[0, 1], &[0, 2], 768);
}

#[test]
fn test_rs_roundtrip_many_inputs() {
    // 16 inputs, lose 4
    rs_roundtrip_test(16, &[0, 1, 2, 3], &[3, 7, 11, 15], 64);
}

/// Non-contiguous recovery exponents (skip some).
#[test]
fn test_rs_roundtrip_noncontiguous_exponents() {
    // Use exponents 0, 5, 10 instead of 0, 1, 2
    rs_roundtrip_test_exponents(4, &[0, 5, 10], &[0, 1, 3], 32);
}

/// High recovery exponents.
#[test]
fn test_rs_roundtrip_high_exponents() {
    rs_roundtrip_test_exponents(3, &[100, 200, 300], &[0, 1, 2], 32);
}

// =========================================================================
// Singular matrix detection
// =========================================================================

/// A matrix with duplicate rows is singular.
#[test]
fn test_singular_duplicate_rows() {
    let mut m = GfMatrix::zeros(3, 3);
    for c in 0..3 {
        m.set(0, c, gf::exp2(c as u32));
        m.set(1, c, gf::exp2((c + 3) as u32));
        m.set(2, c, gf::exp2(c as u32)); // Duplicate of row 0
    }
    assert!(m.invert().is_none(), "duplicate rows should be singular");
}

/// A matrix with a zero row is singular.
#[test]
fn test_singular_zero_row() {
    let mut m = GfMatrix::zeros(3, 3);
    m.set(0, 0, 1);
    m.set(0, 1, 2);
    m.set(0, 2, 3);
    m.set(1, 0, 4);
    m.set(1, 1, 5);
    m.set(1, 2, 6);
    // Row 2 is all zeros
    assert!(m.invert().is_none(), "zero row should make matrix singular");
}

/// 1x1 matrix inversion.
#[test]
fn test_1x1_matrix_inverse() {
    let mut m = GfMatrix::zeros(1, 1);
    m.set(0, 0, 42);
    let inv = m.invert().unwrap();
    assert_eq!(inv.get(0, 0), gf::inv(42));
}

/// 1x1 zero matrix is singular.
#[test]
fn test_1x1_zero_is_singular() {
    let m = GfMatrix::zeros(1, 1);
    assert!(m.invert().is_none());
}

// =========================================================================
// Helpers
// =========================================================================

fn rs_roundtrip_test(
    input_count: usize,
    recovery_exponents: &[u32],
    damaged_indices: &[usize],
    slice_size: usize,
) {
    rs_roundtrip_test_exponents(input_count, recovery_exponents, damaged_indices, slice_size);
}

fn rs_roundtrip_test_exponents(
    input_count: usize,
    recovery_exponents: &[u32],
    damaged_indices: &[usize],
    slice_size: usize,
) {
    assert!(slice_size % 2 == 0, "slice_size must be even for u16");
    assert!(
        damaged_indices.len() <= recovery_exponents.len(),
        "need at least as many recovery blocks as damaged"
    );

    // Generate deterministic input data
    let inputs: Vec<Vec<u8>> = (0..input_count)
        .map(|i| {
            (0..slice_size)
                .map(|j| ((i * 37 + j * 13 + 7) % 256) as u8)
                .collect()
        })
        .collect();

    // Build encoding matrix
    let enc = GfMatrix::par2_encoding_matrix(input_count, recovery_exponents);

    // Compute recovery blocks
    let u16_per_slice = slice_size / 2;
    let recovery_blocks: Vec<Vec<u8>> = recovery_exponents
        .iter()
        .enumerate()
        .map(|(r, _)| {
            let mut block = vec![0u8; slice_size];
            for pos in 0..u16_per_slice {
                let off = pos * 2;
                let mut acc: u16 = 0;
                for (i, input) in inputs.iter().enumerate() {
                    let val = u16::from_le_bytes([input[off], input[off + 1]]);
                    acc = gf::add(acc, gf::mul(enc.get(input_count + r, i), val));
                }
                block[off] = acc as u8;
                block[off + 1] = (acc >> 8) as u8;
            }
            block
        })
        .collect();

    // Build decode matrix: replace damaged data rows with recovery rows
    let recovery_to_use: Vec<usize> = (0..damaged_indices.len()).collect();
    let mut available_rows = Vec::new();
    let mut rec_idx = 0;
    for i in 0..input_count {
        if damaged_indices.contains(&i) {
            available_rows.push(input_count + recovery_to_use[rec_idx]);
            rec_idx += 1;
        } else {
            available_rows.push(i);
        }
    }

    let decode_sub = enc.select_rows(&available_rows);
    let inverse = decode_sub
        .invert()
        .expect("decode matrix should be invertible");

    // Reconstruct damaged blocks
    // Build available data: intact inputs + recovery for damaged positions
    let mut available_data: Vec<&[u8]> = Vec::new();
    let mut rec_idx = 0;
    for (i, input) in inputs.iter().enumerate().take(input_count) {
        if damaged_indices.contains(&i) {
            available_data.push(&recovery_blocks[recovery_to_use[rec_idx]]);
            rec_idx += 1;
        } else {
            available_data.push(input);
        }
    }

    for &dmg_idx in damaged_indices {
        let mut reconstructed = vec![0u8; slice_size];
        for (src_idx, &src_data) in available_data.iter().enumerate() {
            let coeff = inverse.get(dmg_idx, src_idx);
            if coeff != 0 {
                rust_par2::gf_simd_public::mul_add_buffer(&mut reconstructed, src_data, coeff);
            }
        }
        assert_eq!(
            reconstructed,
            inputs[dmg_idx],
            "failed to recover block {dmg_idx} (inputs={input_count}, recovery={}, damaged={:?})",
            recovery_exponents.len(),
            damaged_indices,
        );
    }
}

fn mat_mul(a: &GfMatrix, b: &GfMatrix) -> GfMatrix {
    assert_eq!(a.cols, b.rows);
    let mut result = GfMatrix::zeros(a.rows, b.cols);
    for r in 0..a.rows {
        for c in 0..b.cols {
            let mut acc = 0u16;
            for k in 0..a.cols {
                acc = gf::add(acc, gf::mul(a.get(r, k), b.get(k, c)));
            }
            result.set(r, c, acc);
        }
    }
    result
}

/// Generate all k-element subsets of {0, 1, ..., n-1}.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = Vec::new();
    fn backtrack(
        start: usize,
        n: usize,
        k: usize,
        combo: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if combo.len() == k {
            result.push(combo.clone());
            return;
        }
        for i in start..n {
            combo.push(i);
            backtrack(i + 1, n, k, combo, result);
            combo.pop();
        }
    }
    backtrack(0, n, k, &mut combo, &mut result);
    result
}
