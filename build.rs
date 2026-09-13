//! Points `include_bytes!` at the network file. Set `EVALFILE` to build with another net.
//!
//! The network is not in this repository, since each one is tens of megabytes and every
//! release would add that to the history for good. It is published alongside the engine
//! instead; the error below says where to get it.

const NET_RELEASE: &str = "https://github.com/FirePlank/Peras-networks/releases/download/peras-v3/peras-v3.nnue";

fn main() {
    println!("cargo:rerun-if-env-changed=EVALFILE");
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let default = format!("{root}/nets/peras.nnue");
    let from_env = std::env::var("EVALFILE").ok();
    let path = from_env.clone().unwrap_or_else(|| default.clone());

    if !std::path::Path::new(&path).exists() {
        let what = if from_env.is_some() { "EVALFILE points at" } else { "the default network" };
        panic!(
            "\n\n{what} `{path}`, which does not exist.\n\n\
             Download it into place:\n\n    \
             curl -sL {NET_RELEASE} -o nets/peras.nnue\n\n\
             Or point EVALFILE at a network you already have. Every network the engine has\n\
             shipped is at https://github.com/FirePlank/Peras-networks/releases\n\n"
        );
    }

    println!("cargo:rerun-if-changed={path}");
    println!("cargo:rustc-env=PERAS_NET={path}");
}
