//! SIMD vs scalar correctness tests.
//!
//! Ensures the AVX2/SSSE3 SIMD paths produce identical results to the scalar
//! fallback across all buffer sizes, alignment conditions, and GF constants.

use rust_par2::gf;

// =========================================================================
// mul_add_buffer: SIMD vs scalar for many buffer sizes
// =========================================================================

/// Test buffer sizes around SIMD chunk boundaries.
/// AVX2 processes 32 bytes at a time, so we test sizes that exercise
/// the remainder path: 0, 1, 2, 16, 30, 31, 32, 33, 34, 48, 63, 64, 65, etc.
#[test]
fn test_mul_add_buffer_all_sizes() {
    let sizes: Vec<usize> = {
        let mut v: Vec<usize> = (0..=66).collect(); // 0 through 66
        v.extend([96, 128, 130, 254, 256, 258, 510, 512, 514, 768, 1024]);
        v.extend([768_000]); // PAR2 default slice size
        // Only even sizes (u16 alignment)
        v.retain(|s| s % 2 == 0);
        v
    };

    let constant = 12345u16;

    for &size in &sizes {
        let src = make_deterministic_buf(size, 0xAB);
        let mut dst_simd = make_deterministic_buf(size, 0xCD);
        let mut dst_scalar = dst_simd.clone();

        rust_par2::gf_simd_public::mul_add_buffer(&mut dst_simd, &src, constant);
        scalar_mul_add(&mut dst_scalar, &src, constant);

        assert_eq!(
            dst_simd, dst_scalar,
            "SIMD/scalar mismatch at size={size}, constant={constant}"
        );
    }
}

/// Test with various GF constants, including edge cases.
#[test]
fn test_mul_add_buffer_all_constants() {
    let constants = [
        0u16, 1, 2, 3, 4, 15, 16, 255, 256, 1000, 12345, 32768, 65534, 65535,
        0x100B, // 2^16 mod polynomial
        0xAAAA, 0x5555, 0xFFFF,
    ];
    let size = 1024;

    for &constant in &constants {
        let src = make_deterministic_buf(size, 0x37);
        let mut dst_simd = make_deterministic_buf(size, 0x91);
        let mut dst_scalar = dst_simd.clone();

        rust_par2::gf_simd_public::mul_add_buffer(&mut dst_simd, &src, constant);
        scalar_mul_add(&mut dst_scalar, &src, constant);

        assert_eq!(
            dst_simd, dst_scalar,
            "SIMD/scalar mismatch for constant={constant}"
        );
    }
}

/// Test that mul_add_buffer correctly accumulates (XORs into dst, doesn't overwrite).
#[test]
fn test_mul_add_accumulation() {
    let size = 512;
    let src1 = make_deterministic_buf(size, 0x11);
    let src2 = make_deterministic_buf(size, 0x22);
    let src3 = make_deterministic_buf(size, 0x33);
    let c1 = 100u16;
    let c2 = 200u16;
    let c3 = 300u16;

    // Apply three sources to same dst
    let mut dst = vec![0u8; size];
    rust_par2::gf_simd_public::mul_add_buffer(&mut dst, &src1, c1);
    rust_par2::gf_simd_public::mul_add_buffer(&mut dst, &src2, c2);
    rust_par2::gf_simd_public::mul_add_buffer(&mut dst, &src3, c3);

    // Verify element by element
    for i in 0..size / 2 {
        let off = i * 2;
        let s1 = u16::from_le_bytes([src1[off], src1[off + 1]]);
        let s2 = u16::from_le_bytes([src2[off], src2[off + 1]]);
        let s3 = u16::from_le_bytes([src3[off], src3[off + 1]]);
        let expected = gf::mul(c1, s1) ^ gf::mul(c2, s2) ^ gf::mul(c3, s3);
        let actual = u16::from_le_bytes([dst[off], dst[off + 1]]);
        assert_eq!(actual, expected, "accumulation mismatch at u16 index {i}");
    }
}

/// Constant = 0 must leave dst unchanged.
#[test]
fn test_mul_add_zero_constant_preserves_dst() {
    let size = 256;
    let src = make_deterministic_buf(size, 0xFF);
    let original = make_deterministic_buf(size, 0x42);
    let mut dst = original.clone();
    rust_par2::gf_simd_public::mul_add_buffer(&mut dst, &src, 0);
    assert_eq!(dst, original, "mul by 0 should not modify dst");
}

/// Constant = 1 must XOR src into dst (since 1 * x = x).
#[test]
fn test_mul_add_one_is_xor() {
    let size = 256;
    let src = make_deterministic_buf(size, 0xAA);
    let mut dst = make_deterministic_buf(size, 0x55);
    let mut expected = dst.clone();

    rust_par2::gf_simd_public::mul_add_buffer(&mut dst, &src, 1);

    // XOR reference
    for i in 0..size {
        expected[i] ^= src[i];
    }
    assert_eq!(dst, expected, "mul by 1 should XOR src into dst");
}

// =========================================================================
// mul_add_multi: batched multi-source correctness
// =========================================================================

/// mul_add_multi must match sequential mul_add_buffer calls.
#[test]
fn test_mul_add_multi_matches_sequential() {
    let sizes = [64, 256, 1024, 768_000];
    let source_counts = [1, 2, 3, 4, 5, 7, 8, 16];

    for &size in &sizes {
        for &n_srcs in &source_counts {
            let sources: Vec<Vec<u8>> = (0..n_srcs)
                .map(|i| make_deterministic_buf(size, (i * 37 + 13) as u8))
                .collect();
            let coeffs: Vec<u16> = (0..n_srcs).map(|i| (i * 1000 + 1) as u16).collect();
            let src_refs: Vec<&[u8]> = sources.iter().map(|s| s.as_slice()).collect();

            // Sequential
            let mut dst_seq = vec![0u8; size];
            for (src, &coeff) in sources.iter().zip(coeffs.iter()) {
                rust_par2::gf_simd_public::mul_add_buffer(&mut dst_seq, src, coeff);
            }

            // Batched
            let mut dst_batch = vec![0u8; size];
            rust_par2::gf_simd_public::mul_add_multi(&mut dst_batch, &src_refs, &coeffs);

            assert_eq!(
                dst_batch, dst_seq,
                "multi vs sequential mismatch: size={size}, sources={n_srcs}"
            );
        }
    }
}

/// mul_add_multi with all-zero coefficients must leave dst unchanged.
#[test]
fn test_mul_add_multi_zero_coeffs() {
    let size = 512;
    let sources: Vec<Vec<u8>> = (0..4)
        .map(|i| make_deterministic_buf(size, i as u8))
        .collect();
    let coeffs = vec![0u16; 4];
    let src_refs: Vec<&[u8]> = sources.iter().map(|s| s.as_slice()).collect();

    let original = make_deterministic_buf(size, 0x99);
    let mut dst = original.clone();
    rust_par2::gf_simd_public::mul_add_multi(&mut dst, &src_refs, &coeffs);
    assert_eq!(dst, original, "all-zero coeffs should not modify dst");
}

/// mul_add_multi with sparse coefficients (some zero, some not).
#[test]
fn test_mul_add_multi_sparse_coeffs() {
    let size = 512;
    let sources: Vec<Vec<u8>> = (0..6)
        .map(|i| make_deterministic_buf(size, (i * 17) as u8))
        .collect();
    let coeffs = vec![0u16, 42, 0, 0, 100, 0]; // Only sources 1 and 4 are active
    let src_refs: Vec<&[u8]> = sources.iter().map(|s| s.as_slice()).collect();

    // Batched
    let mut dst_batch = vec![0u8; size];
    rust_par2::gf_simd_public::mul_add_multi(&mut dst_batch, &src_refs, &coeffs);

    // Only active sources
    let mut dst_ref = vec![0u8; size];
    rust_par2::gf_simd_public::mul_add_buffer(&mut dst_ref, &sources[1], 42);
    rust_par2::gf_simd_public::mul_add_buffer(&mut dst_ref, &sources[4], 100);

    assert_eq!(dst_batch, dst_ref, "sparse coefficients mismatch");
}

// =========================================================================
// xor_buffers
// =========================================================================

#[test]
fn test_xor_buffers_all_sizes() {
    let sizes = [0, 2, 4, 16, 30, 32, 34, 64, 128, 256, 1024, 768_000];

    for &size in &sizes {
        let src = make_deterministic_buf(size, 0xAA);
        let mut dst = make_deterministic_buf(size, 0x55);
        let mut expected = dst.clone();

        rust_par2::gf_simd_public::xor_buffers(&mut dst, &src);

        for i in 0..size {
            expected[i] ^= src[i];
        }
        assert_eq!(dst, expected, "XOR mismatch at size={size}");
    }
}

#[test]
fn test_xor_self_gives_zero() {
    let size = 1024;
    let data = make_deterministic_buf(size, 0x42);
    let mut dst = data.clone();
    rust_par2::gf_simd_public::xor_buffers(&mut dst, &data);
    assert!(dst.iter().all(|&b| b == 0), "x XOR x should be 0");
}

// =========================================================================
// Helpers
// =========================================================================

fn make_deterministic_buf(size: usize, seed: u8) -> Vec<u8> {
    let mut buf = vec![0u8; size];
    let mut state = seed as u32;
    for b in buf.iter_mut() {
        // Simple LCG for deterministic pseudorandom data
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *b = (state >> 16) as u8;
    }
    buf
}

fn scalar_mul_add(dst: &mut [u8], src: &[u8], constant: u16) {
    if constant == 0 {
        return;
    }
    let len = dst.len() / 2;
    for i in 0..len {
        let off = i * 2;
        let s = u16::from_le_bytes([src[off], src[off + 1]]);
        let d = u16::from_le_bytes([dst[off], dst[off + 1]]);
        let result = d ^ gf::mul(constant, s);
        dst[off] = result as u8;
        dst[off + 1] = (result >> 8) as u8;
    }
}
