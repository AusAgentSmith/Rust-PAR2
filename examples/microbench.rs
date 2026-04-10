//! Micro-benchmarks to identify hotspots in the GF multiply-accumulate pipeline.
//!
//! Usage: cargo run --release --example microbench
//!
//! Measures:
//!   1. XOR throughput (memory bandwidth ceiling)
//!   2. Single-source GF(2^16) mul_add throughput
//!   3. Multi-source accumulate (simulates repair inner loop)
//!   4. Repair simulation with realistic parameters
//!
//! Reports GB/s and GF-muls/sec so you can compare against theoretical peak.

use std::time::{Duration, Instant};

fn main() {
    println!("=== rust-par2 micro-benchmarks ===\n");

    // Detect CPU features
    #[cfg(target_arch = "x86_64")]
    {
        println!("CPU features:");
        println!("  AVX2:  {}", is_x86_feature_detected!("avx2"));
        println!("  SSSE3: {}", is_x86_feature_detected!("ssse3"));
        println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
        println!("  PCLMULQDQ: {}", is_x86_feature_detected!("pclmulqdq"));
        println!();
    }

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("Available parallelism: {cores} threads\n");

    // Warm up GF tables
    rust_par2::gf_simd_public::mul_add_buffer(&mut [0u8; 4], &[0u8; 4], 1);

    // --- Benchmark parameters ---
    let slice_size = 768_000; // PAR2 default block size (768 KB)
    let large_buf = 4 * 1024 * 1024; // 4 MB for throughput tests

    bench_xor_throughput(large_buf);
    bench_single_mul_add(large_buf);
    bench_single_mul_add(slice_size);
    bench_multi_source_accumulate(slice_size, 16);
    bench_multi_source_accumulate(slice_size, 64);
    bench_multi_source_accumulate(slice_size, 128);
    bench_multi_source_accumulate(slice_size, 512);
    bench_multi_source_accumulate(slice_size, 1365); // ~1GB / 768KB

    println!("\n--- Repair simulation (parallel, rayon) ---\n");
    bench_repair_simulation(slice_size, 1365, 40); // 1GB data, 3% = ~40 damaged
    bench_repair_simulation(slice_size, 1365, 10); // 1GB data, ~0.7% damaged
}

/// Pure XOR — establishes the memory bandwidth ceiling.
fn bench_xor_throughput(buf_size: usize) {
    let src = vec![0xAAu8; buf_size];
    let mut dst = vec![0x55u8; buf_size];

    let (iters, elapsed) = bench_loop(Duration::from_secs(2), || {
        rust_par2::gf_simd_public::xor_buffers(&mut dst, &src);
    });

    let total_bytes = buf_size as f64 * iters as f64;
    let gbps = total_bytes / elapsed.as_secs_f64() / 1e9;
    // Each iteration reads src + reads dst + writes dst = 3 × buf_size
    let mem_gbps = 3.0 * total_bytes / elapsed.as_secs_f64() / 1e9;
    println!("XOR throughput ({}):", fmt_size(buf_size));
    println!("  {gbps:.2} GB/s data, {mem_gbps:.2} GB/s memory traffic");
    println!("  {iters} iterations in {:.3}s", elapsed.as_secs_f64());
    println!();
}

/// Single constant × buffer — measures raw SIMD GF multiply speed.
fn bench_single_mul_add(buf_size: usize) {
    let src = random_buf(buf_size);
    let mut dst = vec![0u8; buf_size];
    let constant = 12345u16;

    let (iters, elapsed) = bench_loop(Duration::from_secs(2), || {
        rust_par2::gf_simd_public::mul_add_buffer(&mut dst, &src, constant);
    });

    let total_bytes = buf_size as f64 * iters as f64;
    let gbps = total_bytes / elapsed.as_secs_f64() / 1e9;
    let gf_muls = (buf_size / 2) as f64 * iters as f64; // u16 elements
    let gf_muls_per_sec = gf_muls / elapsed.as_secs_f64();
    // Memory: read src + read dst + write dst = 3 × buf_size
    let mem_gbps = 3.0 * total_bytes / elapsed.as_secs_f64() / 1e9;
    println!("Single mul_add ({}):", fmt_size(buf_size));
    println!("  {gbps:.2} GB/s data, {mem_gbps:.2} GB/s memory traffic");
    println!("  {:.2} billion GF-muls/sec", gf_muls_per_sec / 1e9);
    println!("  {iters} iterations in {:.3}s", elapsed.as_secs_f64());
    println!();
}

/// Multi-source accumulate: dst ^= Σ c[i] * src[i] — the repair inner loop.
fn bench_multi_source_accumulate(slice_size: usize, num_sources: usize) {
    let sources: Vec<Vec<u8>> = (0..num_sources).map(|_| random_buf(slice_size)).collect();
    let coeffs: Vec<u16> = (1..=num_sources as u16).collect();
    let mut dst = vec![0u8; slice_size];

    // Sequential approach (what repair currently does per damaged block)
    let (iters, elapsed) = bench_loop(Duration::from_secs(2), || {
        dst.fill(0);
        for (src, &coeff) in sources.iter().zip(coeffs.iter()) {
            rust_par2::gf_simd_public::mul_add_buffer(&mut dst, src, coeff);
        }
    });

    let total_data = slice_size as f64 * num_sources as f64 * iters as f64;
    let gbps = total_data / elapsed.as_secs_f64() / 1e9;
    let gf_muls = (slice_size / 2) as f64 * num_sources as f64 * iters as f64;
    let gf_muls_per_sec = gf_muls / elapsed.as_secs_f64();
    // Memory per iteration: num_sources × (read src + read dst + write dst)
    // But dst stays hot in cache after first source, so realistically:
    //   num_sources × read_src + ~2 × slice_size (dst read/write, cached)
    let mem_cold =
        (num_sources as f64 * slice_size as f64 + 2.0 * slice_size as f64) * iters as f64;
    let mem_gbps = mem_cold / elapsed.as_secs_f64() / 1e9;
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    println!(
        "Multi-source accumulate ({} srcs × {}):",
        num_sources,
        fmt_size(slice_size)
    );
    println!("  {gbps:.2} GB/s src data, ~{mem_gbps:.2} GB/s estimated mem traffic");
    println!("  {:.2} billion GF-muls/sec", gf_muls_per_sec / 1e9);
    println!(
        "  {per_iter_ms:.3} ms per accumulation ({iters} iters in {:.3}s)",
        elapsed.as_secs_f64()
    );
    println!();
}

/// Simulates the full repair inner loop with rayon parallelism.
/// num_sources total blocks, num_damaged blocks to reconstruct.
fn bench_repair_simulation(slice_size: usize, num_sources: usize, num_damaged: usize) {
    use rayon::prelude::*;

    let sources: Vec<Vec<u8>> = (0..num_sources).map(|_| random_buf(slice_size)).collect();
    // Build a coefficient matrix [num_damaged × num_sources]
    let coeffs: Vec<Vec<u16>> = (0..num_damaged)
        .map(|d| {
            (0..num_sources)
                .map(|s| ((d * 31 + s * 17 + 1) % 65535) as u16)
                .collect()
        })
        .collect();

    let src_refs: Vec<&[u8]> = sources.iter().map(|s| s.as_slice()).collect();

    // Time the parallel repair loop
    let t = Instant::now();
    let _results: Vec<Vec<u8>> = (0..num_damaged)
        .into_par_iter()
        .map(|dmg_i| {
            let mut dst = vec![0u8; slice_size];
            for (src_idx, src_data) in src_refs.iter().enumerate() {
                let coeff = coeffs[dmg_i][src_idx];
                if coeff != 0 {
                    rust_par2::gf_simd_public::mul_add_buffer(&mut dst, src_data, coeff);
                }
            }
            dst
        })
        .collect();
    let elapsed = t.elapsed();

    let total_src_data = slice_size as f64 * num_sources as f64 * num_damaged as f64;
    let gbps = total_src_data / elapsed.as_secs_f64() / 1e9;
    let gf_muls = (slice_size / 2) as f64 * num_sources as f64 * num_damaged as f64;
    let total_data_gb = num_sources as f64 * slice_size as f64 / 1e9;

    println!(
        "Repair sim: {} srcs × {} damaged × {} = {:.1} GB total data",
        num_sources,
        num_damaged,
        fmt_size(slice_size),
        total_data_gb,
    );
    println!("  {:.3}s elapsed", elapsed.as_secs_f64());
    println!("  {gbps:.2} GB/s src throughput (parallel)");
    println!(
        "  {:.2} billion GF-muls/sec (parallel)",
        gf_muls / elapsed.as_secs_f64() / 1e9
    );
    println!(
        "  {:.1} ms per damaged block",
        elapsed.as_secs_f64() * 1000.0 / num_damaged as f64,
    );
    println!();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run `f` repeatedly for at least `min_duration`, return (iterations, elapsed).
fn bench_loop<F: FnMut()>(min_duration: Duration, mut f: F) -> (u64, Duration) {
    // Warmup
    for _ in 0..3 {
        f();
    }

    let mut iters = 0u64;
    let start = Instant::now();
    loop {
        f();
        iters += 1;
        let elapsed = start.elapsed();
        if elapsed >= min_duration && iters >= 4 {
            return (iters, elapsed);
        }
    }
}

fn random_buf(size: usize) -> Vec<u8> {
    use rand::Rng;
    let mut buf = vec![0u8; size];
    rand::rng().fill_bytes(&mut buf);
    buf
}

fn fmt_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{bytes} B")
    }
}
