#!/bin/bash
# Build and run the I2_S kernel correctness test (utils/i2s_kernel_test/i2s_kernel_test.cpp).
#
#   utils/i2s_kernel_test/run.sh --build-dir <cmake build dir with STATIC ggml libs> [--out <dir>] [--mad]
#
# Always runs the "linked" variant (libggml-cpu from the build dir: the code llama-cli executes).
# With --mad it additionally compiles src/ggml-bitnet-mad.cpp + ggml-cpu-i2s.c straight into the
# test (the kernels patched by upstream microsoft/BitNet PRs #586/#633, which are not part of the
# default CMake build), once with dotprod and once with plain NEON.
# Exit status is non-zero if any test fails. No model file is needed.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD=""; OUT="$ROOT/build-i2s-kernel-test"; MAD=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --build-dir) BUILD="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --mad) MAD=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[[ -d "$BUILD" ]] || { echo "need --build-dir <cmake build dir>" >&2; exit 2; }
mkdir -p "$OUT"
L="$ROOT/3rdparty/llama.cpp"
INC=(-I"$ROOT/include" -I"$L/ggml/include" -I"$L/ggml/src" -I"$L/ggml/src/ggml-cpu")
CXX="${CXX:-clang++}"; CC="${CC:-clang}"
rc=0

LIBS=$(find "$BUILD/3rdparty/llama.cpp/ggml/src" -name 'libggml*.a' | tr '\n' ' ')
[[ -n "$LIBS" ]] || { echo "no static libggml*.a under $BUILD (configure with BUILD_SHARED_LIBS=OFF)" >&2; exit 2; }
FW=""; [[ "$(uname)" == Darwin ]] && FW="-framework Accelerate -framework Metal -framework Foundation -framework MetalKit"
echo "== linked variant ($BUILD)"
$CXX -O2 -std=c++17 "${INC[@]}" "$HERE/i2s_kernel_test.cpp" $LIBS $FW -lpthread -o "$OUT/i2s_kernel_test_linked" || exit 2
"$OUT/i2s_kernel_test_linked" --json "$OUT/linked.json" > "$OUT/linked.txt" || rc=1
cat "$OUT/linked.txt"

if [[ $MAD == 1 ]]; then
  for variant in "dotprod:-mcpu=native+dotprod" "neon-nodotprod:-mcpu=apple-m1+nodotprod"; do
    name=${variant%%:*}; flags=${variant#*:}
    echo; echo "== mad variant: $name ($flags)"
    $CC  -O2 -std=c11 $flags "${INC[@]}" -c "$L/ggml/src/ggml-cpu/ggml-cpu-i2s.c" -o "$OUT/i2s_$name.o" || exit 2
    $CXX -O2 -std=c++17 $flags "${INC[@]}" -c "$ROOT/src/ggml-bitnet-mad.cpp" -o "$OUT/mad_$name.o" || exit 2
    $CXX -O2 -std=c++17 $flags -DTEST_MAD "${INC[@]}" "$HERE/i2s_kernel_test.cpp" "$OUT/i2s_$name.o" "$OUT/mad_$name.o" \
         "$BUILD/3rdparty/llama.cpp/ggml/src/libggml-base.a" -lpthread -o "$OUT/i2s_kernel_test_mad_$name" || exit 2
    "$OUT/i2s_kernel_test_mad_$name" --json "$OUT/mad_$name.json" > "$OUT/mad_$name.txt" || rc=1
    cat "$OUT/mad_$name.txt"
  done
fi
exit $rc
