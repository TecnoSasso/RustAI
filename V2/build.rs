use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell cargo to rerun this script if the kernel changes
    println!("cargo:rerun-if-changed=kernels/matrix_operation.cu");
    
    // Create target directory if it doesn't exist
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = PathBuf::from("target");
    std::fs::create_dir_all(&target_dir).unwrap();
    
    // Path to the output PTX file
    let ptx_output = target_dir.join("matrix_operation.ptx");
    
    // Compile CUDA to PTX
    let status = Command::new("nvcc")
        .args(&[
            "--ptx",
            "-arch=compute_86",  // Usa compute_86 invece di sm_86
            "-code=sm_86",       // Aggiungi questa riga
            "-o", ptx_output.to_str().unwrap(),
            "kernels/matrix_operation.cu"
        ])
        .status()
        .expect("Failed to execute nvcc. Make sure CUDA toolkit is installed and in PATH");
    
    if !status.success() {
        panic!("Failed to compile CUDA kernel to PTX");
    }
    
    println!("cargo:warning=CUDA kernel compiled successfully to PTX");
}
