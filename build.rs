//! Points `include_bytes!` at the network file. Set `EVALFILE` to build with another net.

fn main() {
    println!("cargo:rerun-if-env-changed=EVALFILE");
    let path = std::env::var("EVALFILE")
        .unwrap_or_else(|_| format!("{}/nets/peras.nnue", std::env::var("CARGO_MANIFEST_DIR").unwrap()));
    println!("cargo:rerun-if-changed={path}");
    println!("cargo:rustc-env=PERAS_NET={path}");
}
