#!/bin/bash
#
# setup_llama.sh — Phase 8+10: build the llama.cpp shim (libnmpllama.dylib)
#
# NeuraMeshProtocol never links llama.cpp at build time (the package stays
# dependency-free); instead the Swift runtime dlopens the shim built here.
# The shim compiles against the INSTALLED llama.h, so its struct layouts
# always match the installed library.
#
# Prerequisite:  brew install llama.cpp
# Usage:         scripts/setup_llama.sh
# Output:        Vendor/llama/libnmpllama.dylib
#
# The Swift side finds the dylib via (in order):
#   $NMP_LLAMA_LIB, ./Vendor/llama/libnmpllama.dylib, ~/.nmp/libnmpllama.dylib

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHIM_SOURCE="$REPO_ROOT/scripts/llama-shim/nmp_llama_shim.c"
OUTPUT_DIR="$REPO_ROOT/Vendor/llama"
OUTPUT="$OUTPUT_DIR/libnmpllama.dylib"

if ! command -v brew >/dev/null; then
    echo "error: Homebrew not found — install llama.cpp another way and adapt the paths below" >&2
    exit 1
fi
if ! LLAMA_PREFIX="$(brew --prefix llama.cpp 2>/dev/null)" || [ ! -d "$LLAMA_PREFIX/lib" ]; then
    echo "error: llama.cpp is not installed. Run:  brew install llama.cpp" >&2
    exit 1
fi
BREW_PREFIX="$(brew --prefix)"

echo "llama.cpp: $LLAMA_PREFIX ($("$LLAMA_PREFIX/bin/llama-cli" --version 2>&1 | head -1))"

mkdir -p "$OUTPUT_DIR"
cc -O2 -std=c11 -fPIC -shared \
    -I"$LLAMA_PREFIX/include" \
    -I"$BREW_PREFIX/include" \
    -L"$LLAMA_PREFIX/lib" -lllama \
    -Wl,-rpath,"$LLAMA_PREFIX/lib" \
    -o "$OUTPUT" \
    "$SHIM_SOURCE"

echo "built: $OUTPUT"
echo "smoke test:"
SWIFT_SMOKE='
import Foundation
guard let lib = dlopen(CommandLine.arguments[1], RTLD_NOW) else {
    fatalError(String(cString: dlerror()))
}
typealias AbiFn = @convention(c) () -> Int32
guard let sym = dlsym(lib, "nmp_llama_abi_version") else { fatalError("missing symbol") }
print("  nmp_llama_abi_version =", unsafeBitCast(sym, to: AbiFn.self)())
'
echo "$SWIFT_SMOKE" | swift - "$OUTPUT"
echo "done — run llama-engine tests with:"
echo "  NMP_LLAMA_MODEL=~/models/<model>.gguf swift test --filter Llama"
