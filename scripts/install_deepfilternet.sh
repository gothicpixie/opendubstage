#!/usr/bin/env bash
# Build the DeepFilterNet 3 CLI with float WAV output for ODS.
set -euo pipefail

revision=978576aa8400552a4ce9730838c635aa30db5e61
install_dir=${1:-${HOME}/.local/bin}
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT

git -c advice.detachedHead=false clone --quiet --depth 1 --branch v0.5.6 \
    https://github.com/Rikorose/DeepFilterNet.git "$build_dir/source"
if [[ $(git -C "$build_dir/source" rev-parse HEAD) != "$revision" ]]; then
    echo "DeepFilterNet v0.5.6 does not match the pinned revision" >&2
    exit 1
fi

# The CLI reads floating-point WAV already. Keep its f32 samples on write, too.
git -C "$build_dir/source" apply <<'PATCH'
diff --git a/libDF/src/wav_utils.rs b/libDF/src/wav_utils.rs
--- a/libDF/src/wav_utils.rs
+++ b/libDF/src/wav_utils.rs
@@ -147,11 +147,11 @@ pub fn write_wav_arr2(path: &str, x: ArrayView2<f32>, sr: u32) -> Result<(), WavUtilsError> {
     let spec = hound::WavSpec {
         channels: x.len_of(Axis(0)) as u16,
         sample_rate: sr,
-        bits_per_sample: 16,
-        sample_format: hound::SampleFormat::Int,
+        bits_per_sample: 32,
+        sample_format: hound::SampleFormat::Float,
     };
     let mut writer = WavWriter::create(path, spec)?;
     for xt in x.axis_iter(Axis(1)) {
         for s in xt.iter() {
-            writer.write_sample((s * i16::MAX as f32) as i16)?;
+            writer.write_sample(*s)?;
         }
PATCH

# Upstream's locked time 0.3.28 fails to compile with Rust 1.80 and newer.
cargo update --manifest-path "$build_dir/source/Cargo.toml" --package time --precise 0.3.36
cargo build --manifest-path "$build_dir/source/Cargo.toml" --locked --release \
    --package deep_filter --bin deep-filter --no-default-features --features bin,tract,wav-utils,default-model \
    --target-dir "$build_dir/target"
mkdir -p "$install_dir"
install -m 755 "$build_dir/target/release/deep-filter" "$install_dir/deep-filter-f32"
printf 'Installed %s/deep-filter-f32\n' "$install_dir"
