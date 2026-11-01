#!/bin/bash
# Build engine_logits against the static libllama/libggml of an existing build tree.
#   utils/i2s_gate/build_engine_logits.sh <build-dir> <out-binary>
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="$1"; OUT="$2"; L="$ROOT/3rdparty/llama.cpp"
LIBS=$(find "$BUILD/3rdparty/llama.cpp" \( -name 'libllama.a' -o -name 'libggml*.a' \) | tr '\n' ' ')
[[ -n "$LIBS" ]] || { echo "no static libs under $BUILD (BUILD_SHARED_LIBS=OFF required)" >&2; exit 2; }
FW=""; [[ "$(uname)" == Darwin ]] && FW="-framework Accelerate -framework Metal -framework Foundation -framework MetalKit"
${CXX:-clang++} -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" "$HERE/engine_logits.cpp" $LIBS $FW -lpthread -o "$OUT"
