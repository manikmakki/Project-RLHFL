#!/usr/bin/env bash
# Build llama.cpp with Intel SYCL backend for Arc Pro B60
# Run after setup-llama-intel.sh: bash ~/build-llama-sycl.sh
set -euo pipefail

LLAMA_DIR="$HOME/llama.cpp"
LOG="$HOME/build-llama-sycl.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================="
echo " llama.cpp SYCL Build"
echo " $(date)"
echo "========================================="

# Source oneAPI environment (required for icpx/icx compilers)
echo
echo "[1/4] Sourcing Intel oneAPI environment..."
# setvars.sh references variables (e.g. OCL_ICD_FILENAMES) that may not be set yet.
# Temporarily disable -u (nounset) so those unbound references don't abort the script.
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
echo "  icpx: $(which icpx)"
echo "  icpx version: $(icpx --version 2>&1 | head -1)"

# Verify GPU is visible before building
echo
echo "[2/4] Verifying GPU visibility..."
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
SYCL_DEVICES=$(sycl-ls 2>/dev/null | grep -i 'level_zero:gpu' || true)
if [[ -z "$SYCL_DEVICES" ]]; then
    echo "  WARNING: No Level Zero GPU found via sycl-ls."
    echo "  Check that you're in the render group (run: groups $USER)"
    echo "  and that the Xe driver is loaded (run: lsmod | grep xe)"
    echo "  If you just added yourself to the render group, log out and back in."
    echo
    read -p "  Continue build anyway? [y/N] " -n 1 -r; echo
    [[ "$REPLY" =~ ^[Yy]$ ]] || exit 1
else
    echo "  Found GPU: $SYCL_DEVICES"
fi

# CMake configure
echo
echo "[3/4] Configuring CMake..."
cd "$LLAMA_DIR"

# Remove stale build directory if it doesn't match our required configuration
if [[ -f build/CMakeCache.txt ]]; then
    CACHE_OK=1
    grep -q 'GGML_SYCL:BOOL=ON'        build/CMakeCache.txt 2>/dev/null || CACHE_OK=0
    grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' build/CMakeCache.txt 2>/dev/null || CACHE_OK=0
    if [[ "$CACHE_OK" == "0" ]]; then
        echo "  Stale or incompatible build directory detected, removing..."
        rm -rf build
    fi
elif [[ -d build ]]; then
    rm -rf build
fi

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_DNN=ON \
    \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    \
    -DGGML_NATIVE=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DGGML_OPENMP=ON \
    \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

echo "  CMake configured."

# Build
echo
echo "[4/4] Building llama.cpp (this takes 10-20 minutes)..."
echo "  Using $(nproc) parallel jobs."
cmake --build build --config Release -j "$(nproc)"

echo
echo "========================================="
echo " Build complete!"
echo "========================================="
echo
echo "Key binaries:"
ls -lh "$LLAMA_DIR/build/bin/llama-"{cli,server,bench,ls-sycl-device} 2>/dev/null || \
    ls -lh "$LLAMA_DIR/build/bin/" | head -20
echo
echo "Verify GPU detection with:"
echo "  source /opt/intel/oneapi/setvars.sh"
echo "  export ONEAPI_DEVICE_SELECTOR=level_zero:0 ZES_ENABLE_SYSMAN=1"
echo "  $LLAMA_DIR/build/bin/llama-ls-sycl-device"
echo
echo "Then see ~/run-llama.sh for example inference commands."
echo "Full log saved to: $LOG"