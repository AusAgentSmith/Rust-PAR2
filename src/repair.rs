//! PAR2 repair engine.
//!
//! Repairs damaged or missing files using Reed-Solomon recovery data.
//!
//! Algorithm (D×D reduced approach — D = damaged blocks, N = total blocks):
//! 1. Verify to identify damaged/missing blocks
//! 2. Load recovery blocks from volume files
//! 3. Build a D×D Vandermonde submatrix (only damaged columns × recovery rows)
//! 4. Invert the D×D matrix — O(D³) instead of O(N²D) for the full N×N
//! 5. Compute adjusted recovery: subtract intact-block contributions from recovery data
//! 6. Apply the D×D inverse to produce repaired blocks
//! 7. Write repaired blocks back to files

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use rayon::prelude::*;
use tracing::{debug, info};

use crate::gf;
use crate::gf_simd;
use crate::matrix::{GfMatrix, par2_input_constants};
use crate::recovery::{RecoveryBlock, load_recovery_blocks};
use crate::types::{Par2FileSet, VerifyResult};
use crate::verify;

/// Result of a repair operation.
#[derive(Debug)]
pub struct RepairResult {
    /// Whether the repair succeeded (all files now intact).
    pub success: bool,
    /// Number of blocks repaired.
    pub blocks_repaired: u32,
    /// Number of files repaired.
    pub files_repaired: usize,
    /// Descriptive message.
    pub message: String,
}

impl fmt::Display for RepairResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(
                f,
                "Repair complete: {} blocks repaired across {} files",
                self.blocks_repaired, self.files_repaired
            )
        } else {
            write!(f, "Repair failed: {}", self.message)
        }
    }
}

/// Errors that can occur during repair.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Insufficient recovery data: need {needed} blocks, have {available}")]
    InsufficientRecovery { needed: u32, available: u32 },
    #[error("Decode matrix is singular — cannot repair with these recovery blocks")]
    SingularMatrix,
    #[error("No damage detected — nothing to repair")]
    NoDamage,
    #[error("Verification after repair failed: {0}")]
    VerifyFailed(String),
}

/// Repair damaged/missing files in a PAR2 set.
///
/// This is a blocking operation. For async contexts, wrap in `spawn_blocking`.
///
/// Runs verification internally to identify damage. If you already have a
/// [`VerifyResult`] from a prior [`verify()`](crate::verify) call, use
/// [`repair_from_verify`] instead to skip the redundant verification pass.
pub fn repair(file_set: &Par2FileSet, dir: &Path) -> Result<RepairResult, RepairError> {
    let verify_result = verify::verify(file_set, dir);
    repair_from_verify_inner(file_set, dir, &verify_result, true)
}

/// Repair using a pre-computed [`VerifyResult`].
///
/// This skips the initial verification pass, saving significant time when the
/// caller has already called [`verify()`](crate::verify). The `verify_result`
/// must have been computed against the same `file_set` and `dir`.
///
/// Uses a reduced D×D matrix approach where D = number of damaged blocks.
/// The full N×N decode matrix is never constructed — only a small Vandermonde
/// submatrix covering damaged positions is built and inverted. This reduces
/// matrix inversion from O(N²D) to O(D³), which is the difference between
/// minutes and milliseconds for typical repair scenarios.
///
/// After repair, a full MD5 verification pass confirms all files are correct.
/// To skip re-verification, use [`repair_from_verify_no_reverify`].
///
/// This is a blocking operation. For async contexts, wrap in `spawn_blocking`.
pub fn repair_from_verify(
    file_set: &Par2FileSet,
    dir: &Path,
    verify_result: &VerifyResult,
) -> Result<RepairResult, RepairError> {
    repair_from_verify_inner(file_set, dir, verify_result, true)
}

/// Like [`repair_from_verify`], but skips re-verification after repair.
///
/// The Reed-Solomon math is deterministic — if the matrix inverted and I/O
/// succeeded, the output is correct. Skipping re-verify saves a full read
/// of all files (~6s per 5GB).
pub fn repair_from_verify_no_reverify(
    file_set: &Par2FileSet,
    dir: &Path,
    verify_result: &VerifyResult,
) -> Result<RepairResult, RepairError> {
    repair_from_verify_inner(file_set, dir, verify_result, false)
}

fn repair_from_verify_inner(
    file_set: &Par2FileSet,
    dir: &Path,
    verify_result: &VerifyResult,
    re_verify: bool,
) -> Result<RepairResult, RepairError> {
    if verify_result.all_correct() {
        return Err(RepairError::NoDamage);
    }

    let blocks_needed = verify_result.blocks_needed();
    info!(
        blocks_needed,
        damaged = verify_result.damaged.len(),
        missing = verify_result.missing.len(),
        "Repair: damage detected"
    );

    // Step 1: Load recovery blocks from all volume files
    let recovery_blocks = load_recovery_blocks(dir, &file_set.recovery_set_id, file_set.slice_size);

    if (recovery_blocks.len() as u32) < blocks_needed {
        return Err(RepairError::InsufficientRecovery {
            needed: blocks_needed,
            available: recovery_blocks.len() as u32,
        });
    }

    // Step 2: Map files to global block indices
    let block_map = build_block_map(file_set);
    let total_input_blocks = block_map.total_blocks as usize;

    let damaged_indices = find_damaged_block_indices(verify_result, &block_map);
    let num_damaged = damaged_indices.len();
    info!(
        damaged_block_count = num_damaged,
        total_input_blocks, "Mapped damaged blocks to global indices"
    );

    // Select exactly D recovery blocks (one per damaged block)
    let recovery_to_use: Vec<&RecoveryBlock> = recovery_blocks.iter().take(num_damaged).collect();
    let recovery_exponents: Vec<u32> = recovery_to_use.iter().map(|b| b.exponent).collect();

    // Step 3: Build and invert the D×D Vandermonde submatrix.
    //
    // The full encoding equation for recovery block e is:
    //   R_e = Σ_i (c_i^exp_e × D_i)   for all input blocks i
    //
    // Rearranging for the damaged blocks only:
    //   R_e - Σ_{intact i} (c_i^exp_e × D_i) = Σ_{damaged j} (c_j^exp_e × D_j)
    //
    // The D×D matrix V[e][j] = c_{damaged_j}^{exp_e} is what we invert.
    let constants = par2_input_constants(total_input_blocks);

    let mut vandermonde = GfMatrix::zeros(num_damaged, num_damaged);
    for (e, &exp) in recovery_exponents.iter().enumerate() {
        for (j, &dmg_idx) in damaged_indices.iter().enumerate() {
            vandermonde.set(e, j, gf::pow(constants[dmg_idx], exp));
        }
    }

    let inverse = vandermonde.invert().ok_or(RepairError::SingularMatrix)?;

    info!(
        "D×D decode matrix inverted ({}×{})",
        num_damaged, num_damaged
    );

    // Step 4: Compute adjusted recovery buffers.
    //
    // adjusted[e] = R_e ⊕ Σ_{intact i} (c_i^exp_e × D_i)
    //
    // In GF(2^16), subtraction = addition = XOR, so we XOR-accumulate the
    // intact-block contributions into the recovery data.
    //
    // Cache-tiled approach: the D output buffers (D × slice_size) typically
    // exceed L3 cache. Iterating source-major (one source at a time against
    // all D outputs) thrashes cache on every source. Instead we read batches
    // of B source blocks, then process each output buffer against the entire
    // batch — keeping the output buffer hot in L1/L2 while applying B sources
    // via mul_add_multi's pair-batched AVX2 path.

    let slice_size = file_set.slice_size as usize;

    // Build a fast lookup for damaged indices
    let damaged_set: std::collections::HashSet<usize> = damaged_indices.iter().copied().collect();

    // Initialize adjusted recovery from the actual recovery block data
    let mut adjusted: Vec<Vec<u8>> = recovery_to_use.iter().map(|rb| rb.data.clone()).collect();

    // Collect intact block indices for batch reading
    let intact_indices: Vec<usize> = (0..total_input_blocks)
        .filter(|i| !damaged_set.contains(i))
        .collect();

    // Batch size: chosen so B × slice_size fits comfortably in L3 cache (~16-32MB).
    // With 768KB slices, 24 blocks = ~18MB. Leaves room for output buffer in L1/L2.
    const BATCH_SIZE: usize = 24;

    // Read and process source blocks in cache-friendly batches.
    // I/O is sequential so OS readahead keeps the pipeline full.
    let mut file_handles: HashMap<String, std::fs::File> = HashMap::new();

    for batch in intact_indices.chunks(BATCH_SIZE) {
        // Read this batch of source blocks
        let batch_data: Vec<Vec<u8>> = batch
            .iter()
            .map(|&idx| read_source_block(dir, &block_map, idx, slice_size, &mut file_handles))
            .collect::<std::io::Result<Vec<_>>>()?;

        let batch_refs: Vec<&[u8]> = batch_data.iter().map(|v| v.as_slice()).collect();

        // For each output buffer (parallelized across cores), apply all sources
        // in this batch at once. The output buffer stays in L1/L2 for the whole
        // batch, and mul_add_multi uses pair-batched AVX2 to halve dst bandwidth.
        adjusted
            .par_iter_mut()
            .enumerate()
            .for_each(|(e, adj_buf)| {
                let coeffs: Vec<u16> = batch
                    .iter()
                    .map(|&src_idx| gf::pow(constants[src_idx], recovery_exponents[e]))
                    .collect();
                gf_simd::mul_add_multi(adj_buf, &batch_refs, &coeffs);
            });
    }

    info!("Intact-block contributions subtracted from recovery data");

    // Step 5: Apply the D×D inverse to produce repaired blocks.
    //
    // repaired[j] = Σ_e (V^{-1}[j][e] × adjusted[e])
    //
    // This is D × D mul-adds of slice_size — small relative to the streaming phase.

    let adj_refs: Vec<&[u8]> = adjusted.iter().map(|v| v.as_slice()).collect();

    let mut outputs: Vec<Vec<u8>> = (0..num_damaged).map(|_| vec![0u8; slice_size]).collect();

    outputs.par_iter_mut().enumerate().for_each(|(j, dst)| {
        let coeffs: Vec<u16> = (0..num_damaged).map(|e| inverse.get(j, e)).collect();
        gf_simd::mul_add_multi(dst, &adj_refs, &coeffs);
    });

    info!("Repaired blocks reconstructed via D×D inverse");

    let repaired_blocks: Vec<(usize, Vec<u8>)> =
        damaged_indices.iter().copied().zip(outputs).collect();

    // Step 6: Write repaired blocks back to files
    let mut files_touched = std::collections::HashSet::new();

    for (global_idx, data) in &repaired_blocks {
        let (filename, file_offset, write_len) = block_map.global_to_file(*global_idx, slice_size);

        let file_path = dir.join(&filename);
        debug!(
            filename,
            global_block = global_idx,
            offset = file_offset,
            len = write_len,
            "Writing repaired block"
        );

        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&file_path)?;

        // Ensure file is the right size (for missing files)
        let expected_size = block_map
            .files
            .iter()
            .find(|bf| bf.filename == filename)
            .map(|bf| bf.file_size)
            .unwrap_or(0);
        let current_size = f.metadata()?.len();
        if current_size < expected_size {
            f.set_len(expected_size)?;
        }

        f.seek(SeekFrom::Start(file_offset as u64))?;
        f.write_all(&data[..write_len])?;
        files_touched.insert(filename.clone());
    }

    // Step 7: Optional re-verify
    if re_verify {
        let verification = verify::verify(file_set, dir);
        if verification.all_correct() {
            info!(
                blocks = repaired_blocks.len(),
                files = files_touched.len(),
                "Repair successful — all files verified"
            );
            Ok(RepairResult {
                success: true,
                blocks_repaired: repaired_blocks.len() as u32,
                files_repaired: files_touched.len(),
                message: "All files repaired and verified".to_string(),
            })
        } else {
            Err(RepairError::VerifyFailed(format!("{verification}")))
        }
    } else {
        info!(
            blocks = repaired_blocks.len(),
            files = files_touched.len(),
            "Repair complete (re-verify skipped)"
        );
        Ok(RepairResult {
            success: true,
            blocks_repaired: repaired_blocks.len() as u32,
            files_repaired: files_touched.len(),
            message: "All files repaired (re-verify skipped)".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Block mapping
// ---------------------------------------------------------------------------

/// Maps between global block indices and per-file block positions.
struct BlockMap {
    files: Vec<BlockFile>,
    total_blocks: u32,
}

struct BlockFile {
    filename: String,
    file_size: u64,
    block_count: u32,
    /// First global block index for this file.
    start_block: u32,
}

fn build_block_map(file_set: &Par2FileSet) -> BlockMap {
    let slice_size = file_set.slice_size;
    let mut files = Vec::new();
    let mut block_offset = 0u32;

    // Sort files by file ID for deterministic ordering (same as par2cmdline)
    let mut sorted_files: Vec<_> = file_set.files.values().collect();
    sorted_files.sort_by_key(|f| f.file_id);

    for f in sorted_files {
        let block_count = if slice_size == 0 {
            0
        } else {
            f.size.div_ceil(slice_size) as u32
        };
        files.push(BlockFile {
            filename: f.filename.clone(),
            file_size: f.size,
            block_count,
            start_block: block_offset,
        });
        block_offset += block_count;
    }

    BlockMap {
        files,
        total_blocks: block_offset,
    }
}

impl BlockMap {
    /// Convert a global block index to (filename, file_byte_offset, bytes_to_write).
    fn global_to_file(&self, global_idx: usize, slice_size: usize) -> (String, usize, usize) {
        let global = global_idx as u32;
        for f in &self.files {
            if global >= f.start_block && global < f.start_block + f.block_count {
                let local_block = (global - f.start_block) as usize;
                let file_offset = local_block * slice_size;
                // Last block may be shorter than slice_size
                let remaining = f.file_size as usize - file_offset;
                let write_len = remaining.min(slice_size);
                return (f.filename.clone(), file_offset, write_len);
            }
        }
        panic!("Global block index {global_idx} out of range");
    }
}

fn find_damaged_block_indices(verify_result: &VerifyResult, block_map: &BlockMap) -> Vec<usize> {
    let mut indices = Vec::new();

    for damaged in &verify_result.damaged {
        if let Some(bf) = block_map
            .files
            .iter()
            .find(|f| f.filename == damaged.filename)
        {
            if damaged.damaged_block_indices.is_empty() {
                // No per-block info — assume all blocks damaged
                for i in 0..bf.block_count {
                    indices.push((bf.start_block + i) as usize);
                }
            } else {
                // Use precise per-block damage info
                for &local_idx in &damaged.damaged_block_indices {
                    indices.push((bf.start_block + local_idx) as usize);
                }
            }
        }
    }

    for missing in &verify_result.missing {
        if let Some(bf) = block_map
            .files
            .iter()
            .find(|f| f.filename == missing.filename)
        {
            for i in 0..bf.block_count {
                indices.push((bf.start_block + i) as usize);
            }
        }
    }

    indices.sort();
    indices.dedup();
    indices
}

/// Read a single source block from disk, reusing file handles.
fn read_source_block(
    dir: &Path,
    block_map: &BlockMap,
    global_idx: usize,
    slice_size: usize,
    file_handles: &mut HashMap<String, std::fs::File>,
) -> std::io::Result<Vec<u8>> {
    let (filename, file_offset, _) = block_map.global_to_file(global_idx, slice_size);

    let handle = match file_handles.entry(filename.clone()) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let path = dir.join(&filename);
            e.insert(std::fs::File::open(&path)?)
        }
    };
    handle.seek(SeekFrom::Start(file_offset as u64))?;

    let mut buf = vec![0u8; slice_size]; // zero-initialized for last-block padding
    let mut total = 0;
    while total < slice_size {
        match handle.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the basic RS encode→decode round-trip works with the D×D approach.
    /// 2 data blocks, 2 recovery blocks, lose both data blocks, recover.
    #[test]
    fn test_rs_roundtrip_simple() {
        // 2 input "blocks" of 4 bytes each (2 u16 values)
        let input0: Vec<u8> = vec![0x01, 0x00, 0x02, 0x00]; // [1, 2] as u16 LE
        let input1: Vec<u8> = vec![0x03, 0x00, 0x04, 0x00]; // [3, 4] as u16 LE

        let input_count = 2;
        let recovery_exponents = vec![0u32, 1u32];

        // Build encoding matrix to compute recovery blocks
        let enc = GfMatrix::par2_encoding_matrix(input_count, &recovery_exponents);

        // Compute recovery blocks
        let slice_size = 4;
        let u16_per_slice = slice_size / 2;
        let inputs = [&input0, &input1];

        let mut recovery0 = vec![0u8; slice_size];
        let mut recovery1 = vec![0u8; slice_size];

        for pos in 0..u16_per_slice {
            let off = pos * 2;
            let mut r0: u16 = 0;
            let mut r1: u16 = 0;
            for (i, inp) in inputs.iter().enumerate() {
                let val = u16::from_le_bytes([inp[off], inp[off + 1]]);
                r0 = gf::add(r0, gf::mul(enc.get(2, i), val));
                r1 = gf::add(r1, gf::mul(enc.get(3, i), val));
            }
            recovery0[off] = r0 as u8;
            recovery0[off + 1] = (r0 >> 8) as u8;
            recovery1[off] = r1 as u8;
            recovery1[off + 1] = (r1 >> 8) as u8;
        }

        // Now "lose" both input blocks. Use D×D approach to recover.
        // D = 2 (both blocks damaged), N = 2.
        // Since all blocks are damaged, there are no intact blocks to subtract.
        // adjusted[e] = recovery[e] (no intact contributions to remove).

        let constants = par2_input_constants(input_count);
        let damaged_indices = [0usize, 1usize];

        // Build D×D Vandermonde: V[e][j] = c_{damaged_j}^{exp_e}
        let num_damaged = damaged_indices.len();
        let mut vandermonde = GfMatrix::zeros(num_damaged, num_damaged);
        for (e, &exp) in recovery_exponents.iter().enumerate() {
            for (j, &dmg_idx) in damaged_indices.iter().enumerate() {
                vandermonde.set(e, j, gf::pow(constants[dmg_idx], exp));
            }
        }

        let inv = vandermonde.invert().expect("Should be invertible");

        // Apply inverse: repaired[j] = Σ_e inv[j][e] * adjusted[e]
        let adjusted = [&recovery0[..], &recovery1[..]];
        let mut result0 = vec![0u8; slice_size];
        let mut result1 = vec![0u8; slice_size];

        for pos in 0..u16_per_slice {
            let off = pos * 2;
            let mut out0: u16 = 0;
            let mut out1: u16 = 0;
            for (e, adj) in adjusted.iter().enumerate() {
                let val = u16::from_le_bytes([adj[off], adj[off + 1]]);
                out0 = gf::add(out0, gf::mul(inv.get(0, e), val));
                out1 = gf::add(out1, gf::mul(inv.get(1, e), val));
            }
            result0[off] = out0 as u8;
            result0[off + 1] = (out0 >> 8) as u8;
            result1[off] = out1 as u8;
            result1[off + 1] = (out1 >> 8) as u8;
        }

        assert_eq!(result0, input0, "Recovered block 0 should match original");
        assert_eq!(result1, input1, "Recovered block 1 should match original");
    }

    /// Test D×D approach with partial damage (some blocks intact).
    /// 4 data blocks, lose 2, recover using 2 recovery blocks.
    #[test]
    fn test_rs_roundtrip_partial_damage() {
        let slice_size = 4;
        let input_count = 4;
        let recovery_exponents = vec![0u32, 1u32];

        let inputs: Vec<Vec<u8>> = vec![
            vec![0x01, 0x00, 0x02, 0x00],
            vec![0x03, 0x00, 0x04, 0x00],
            vec![0x05, 0x00, 0x06, 0x00],
            vec![0x07, 0x00, 0x08, 0x00],
        ];

        let enc = GfMatrix::par2_encoding_matrix(input_count, &recovery_exponents);

        // Compute recovery blocks
        let mut recovery = vec![vec![0u8; slice_size]; 2];
        for pos in 0..(slice_size / 2) {
            let off = pos * 2;
            for (e, rec) in recovery.iter_mut().enumerate() {
                let mut val: u16 = 0;
                for (i, inp) in inputs.iter().enumerate() {
                    let d = u16::from_le_bytes([inp[off], inp[off + 1]]);
                    val = gf::add(val, gf::mul(enc.get(input_count + e, i), d));
                }
                rec[off] = val as u8;
                rec[off + 1] = (val >> 8) as u8;
            }
        }

        // "Lose" blocks 1 and 3. Blocks 0 and 2 are intact.
        let damaged_indices = [1usize, 3usize];
        let intact_indices: Vec<usize> = (0..input_count)
            .filter(|i| !damaged_indices.contains(i))
            .collect();
        let num_damaged = damaged_indices.len();

        let constants = par2_input_constants(input_count);

        // Build D×D Vandermonde
        let mut vandermonde = GfMatrix::zeros(num_damaged, num_damaged);
        for (e, &exp) in recovery_exponents.iter().enumerate() {
            for (j, &dmg_idx) in damaged_indices.iter().enumerate() {
                vandermonde.set(e, j, gf::pow(constants[dmg_idx], exp));
            }
        }
        let inv = vandermonde.invert().expect("Should be invertible");

        // Compute adjusted recovery: subtract intact contributions
        let mut adjusted = recovery.clone();
        for &intact_idx in &intact_indices {
            let c_i = constants[intact_idx];
            for (e, adj) in adjusted.iter_mut().enumerate() {
                let coeff = gf::pow(c_i, recovery_exponents[e]);
                gf_simd::mul_add_buffer(adj, &inputs[intact_idx], coeff);
            }
        }

        // Apply inverse
        let adj_refs: Vec<&[u8]> = adjusted.iter().map(|v| v.as_slice()).collect();
        let mut outputs: Vec<Vec<u8>> = (0..num_damaged).map(|_| vec![0u8; slice_size]).collect();

        for (j, dst) in outputs.iter_mut().enumerate() {
            let coeffs: Vec<u16> = (0..num_damaged).map(|e| inv.get(j, e)).collect();
            gf_simd::mul_add_multi(dst, &adj_refs, &coeffs);
        }

        // Verify recovery
        assert_eq!(outputs[0], inputs[1], "Recovered block 1 should match");
        assert_eq!(outputs[1], inputs[3], "Recovered block 3 should match");
    }
}
