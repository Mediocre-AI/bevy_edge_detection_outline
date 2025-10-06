export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
cargo build --target wasm32-unknown-unknown --release --example 3d_shapes
wasm-bindgen --out-dir target/ --target web target/wasm32-unknown-unknown/release/examples/3d_shapes.wasm