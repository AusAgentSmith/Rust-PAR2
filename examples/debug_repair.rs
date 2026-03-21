fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fixture = args.get(1).map(|s| s.as_str()).unwrap_or("missing");
    let src = format!("tests/fixtures/{fixture}");
    let tmp = tempfile::tempdir().unwrap();

    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
        }
    }

    let par2_path = tmp.path().join("testdata.bin.par2");
    let file_set = rust_par2::parse(&par2_path).unwrap();

    println!(
        "Files: {}, slice_size: {}",
        file_set.files.len(),
        file_set.slice_size
    );
    for f in file_set.files.values() {
        println!(
            "  {} ({} bytes, {} slices, md5: {:02x?})",
            f.filename,
            f.size,
            f.slices.len(),
            &f.hash[..4]
        );
    }

    let verify = rust_par2::verify(&file_set, tmp.path());
    println!("\nVerify: {verify}");

    if verify.all_correct() {
        println!("All correct, nothing to repair.");
        return;
    }

    // Load recovery blocks for debugging
    let recovery = rust_par2::recovery::load_recovery_blocks(
        tmp.path(),
        &file_set.recovery_set_id,
        file_set.slice_size,
    );
    println!("\nRecovery blocks: {}", recovery.len());
    for b in &recovery {
        println!(
            "  exponent={}, data_len={}, first 8 bytes: {:02x?}",
            b.exponent,
            b.data.len(),
            &b.data[..8]
        );
    }

    match rust_par2::repair(&file_set, tmp.path()) {
        Ok(r) => {
            println!("\nRepair: {r}");
            let repaired = tmp.path().join("testdata.bin");
            if repaired.exists() {
                let data = std::fs::read(&repaired).unwrap();
                println!("Repaired file: {} bytes", data.len());

                // Compute MD5
                use md5::{Digest, Md5};
                let hash: [u8; 16] = Md5::digest(&data).into();
                println!("MD5: {:02x?}", hash);
                println!(
                    "Expected: {:02x?}",
                    file_set.files.values().next().unwrap().hash
                );
            }
        }
        Err(e) => println!("\nRepair error: {e}"),
    }
}
