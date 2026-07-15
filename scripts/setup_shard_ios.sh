#!/bin/bash
#
# setup_shard_ios.sh — Phase 10: build the ggml graph-surgery shard shim as an
# iOS xcframework so the NeuraMeshPeer app can do REAL sharded compute on an
# iPhone/iPad (partial-load only its assigned layer range).
#
# Where scripts/setup_shard.sh builds a macOS .dylib against the `ggml` brew
# formula, this builds ggml FROM SOURCE for arm64-ios (device) and the
# simulator, CPU-only (the shim computes on CPU — no Metal shader embedding
# needed), links it with the shim into a DYNAMIC `nmpshard.framework`, and
# wraps both slices in `Vendor/ios/nmpshard.xcframework`.
#
# The Swift binding (LlamaShardRuntime.swift) dlopen's the framework from the
# app bundle at runtime; the app auto-selects the real engine once it is
# present (PeerViewModel.makeEngine) and a .gguf is in Documents.
#
# Prerequisites:  Xcode + command line tools, cmake, git.
# Usage:          scripts/setup_shard_ios.sh            # device + simulator
#                 scripts/setup_shard_ios.sh --sim      # simulator only (fast validate)
#
# AFTER running: in Xcode, add Vendor/ios/nmpshard.xcframework to the
# NeuraMeshPeer target → "Frameworks, Libraries, and Embedded Content" →
# Embed & Sign. Then build to your device with your signing team. On-device
# real compute must be validated on the device (this script only produces the
# framework; it cannot deploy or sign for you).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/scripts/llama-shim/nmp_shard_shim.c"
BUILD="$REPO_ROOT/.build/ios-shard"
OUT_DIR="$REPO_ROOT/Vendor/ios"
GGML_REPO="https://github.com/ggml-org/ggml.git"
# Pin a known-good ggml commit (matches the API the shim uses: gguf_*,
# ggml_backend_*, GGML_BACKEND_DEVICE_TYPE_CPU). Update deliberately.
GGML_REF="${GGML_REF:-master}"
IOS_MIN="15.0"

SIM_ONLY=0
[ "${1:-}" = "--sim" ] && SIM_ONLY=1

command -v cmake >/dev/null || { echo "error: cmake not found (brew install cmake)" >&2; exit 1; }
command -v git   >/dev/null || { echo "error: git not found" >&2; exit 1; }

mkdir -p "$BUILD" "$OUT_DIR"

# --- 1. ggml source ---------------------------------------------------------
if [ ! -d "$BUILD/ggml/.git" ]; then
    echo "[ios-shard] cloning ggml ($GGML_REF) …"
    git clone --depth 1 --branch "$GGML_REF" "$GGML_REPO" "$BUILD/ggml" 2>/dev/null \
        || git clone --depth 1 "$GGML_REPO" "$BUILD/ggml"
fi

# --- 2. build ggml (CPU-only, static) for one Apple platform ----------------
# args: <tag> <CMAKE_SYSTEM_NAME> <sysroot> <arch>
build_ggml() {
    local tag="$1" sysname="$2" sysroot="$3" arch="$4"
    local bdir="$BUILD/ggml-$tag"
    echo "[ios-shard] building ggml for $tag ($arch, $sysroot) …"
    cmake -S "$BUILD/ggml" -B "$bdir" -G Xcode \
        -DCMAKE_SYSTEM_NAME="$sysname" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_SYSROOT="$sysroot" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF >/dev/null
    cmake --build "$bdir" --config Release --target ggml ggml-base ggml-cpu >/dev/null
}

# args: <tag> <clang -target triple> <sysroot>
link_framework() {
    local tag="$1" triple="$2" sysroot="$3"
    local bdir="$BUILD/ggml-$tag"
    local fw="$BUILD/fw-$tag/nmpshard.framework"
    # NOTE: this function returns $fw on stdout (captured by the caller), so ALL
    # human-facing output here MUST go to stderr.
    echo "[ios-shard] linking nmpshard.framework for $tag …" >&2
    rm -rf "$fw"; mkdir -p "$fw"
    # Compile the shim against the ggml headers, static-backends path.
    xcrun clang -c -O2 -target "$triple" -isysroot "$(xcrun --sdk "$sysroot" --show-sdk-path)" \
        -DNMP_STATIC_BACKENDS -I"$BUILD/ggml/include" -I"$BUILD/ggml/src" \
        "$SRC" -o "$BUILD/shim-$tag.o"
    # Collect the ggml static archives cmake produced.
    local archives
    archives=$(find "$bdir" -name 'libggml*.a' | sort -u)
    [ -n "$archives" ] || { echo "error: no libggml*.a for $tag" >&2; exit 1; }
    # Link shim + ggml into ONE dynamic framework binary.
    xcrun clang -dynamiclib -target "$triple" \
        -isysroot "$(xcrun --sdk "$sysroot" --show-sdk-path)" \
        -install_name "@rpath/nmpshard.framework/nmpshard" \
        -Wl,-force_load,"$BUILD/shim-$tag.o" \
        $(printf -- '-Wl,-force_load,%s ' $archives) \
        -lc++ -framework Foundation \
        -o "$fw/nmpshard"
    # Minimal Info.plist so it is a valid, embeddable framework.
    /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.neuramesh.nmpshard" \
        -c "Add :CFBundleName string nmpshard" \
        -c "Add :CFBundleExecutable string nmpshard" \
        -c "Add :CFBundlePackageType string FMWK" \
        -c "Add :MinimumOSVersion string $IOS_MIN" \
        "$fw/Info.plist" >/dev/null
    echo "$fw"
}

FRAMEWORKS=()

# Simulator slice (arm64 — matches Apple Silicon Macs).
build_ggml sim iOS "iphonesimulator" arm64
FRAMEWORKS+=("-framework" "$(link_framework sim arm64-apple-ios${IOS_MIN}-simulator iphonesimulator)")

if [ "$SIM_ONLY" -eq 0 ]; then
    build_ggml dev iOS "iphoneos" arm64
    FRAMEWORKS+=("-framework" "$(link_framework dev arm64-apple-ios${IOS_MIN} iphoneos)")
fi

# --- 3. xcframework ---------------------------------------------------------
rm -rf "$OUT_DIR/nmpshard.xcframework"
xcodebuild -create-xcframework "${FRAMEWORKS[@]}" \
    -output "$OUT_DIR/nmpshard.xcframework" >/dev/null

echo ""
echo "ok: $OUT_DIR/nmpshard.xcframework"
echo "next: in Xcode, add it to NeuraMeshPeer → Frameworks → Embed & Sign,"
echo "      drop a .gguf into the app's Documents, then build to your device."
