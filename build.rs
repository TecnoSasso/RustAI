use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/matrix_operation.cu");

    let target_dir = PathBuf::from("target");
    std::fs::create_dir_all(&target_dir).unwrap();
    let ptx_output = target_dir.join("matrix_operation.ptx");

    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .expect("Failed to run nvidia-smi. Make sure NVIDIA drivers are installed");
    let compute_cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let parts: Vec<&str> = compute_cap.split('.').collect();
    if parts.len() != 2 {
        panic!("Unexpected compute capability format: {}", compute_cap);
    }
    let arch = format!("compute_{}{}", parts[0], parts[1]);
    let code = format!("sm_{}{}", parts[0], parts[1]);

    let status = Command::new("nvcc")
        .args(&[
            "--ptx",
            "-arch", &arch,
            "-code", &code,
            "-o", ptx_output.to_str().unwrap(),
            "kernels/matrix_operation.cu"
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("Failed to compile CUDA kernel to PTX");
    }

    println!("cargo:warning=CUDA kernel compiled successfully to PTX for {}", compute_cap);
}