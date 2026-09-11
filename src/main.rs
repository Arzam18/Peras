fn main() {
    peras::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    peras::uci::run(args);
}
