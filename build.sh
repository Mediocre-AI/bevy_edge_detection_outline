export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen --out-dir target/ --target web target/wasm32-unknown-unknown/release/app.wasm