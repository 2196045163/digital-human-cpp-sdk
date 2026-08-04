#!/usr/bin/env bash
# =============================================================================
# M05 run_performance_benchmark.sh — 全链路性能基准采集
# =============================================================================
#
# 功能：
#   1. 环境信息采集（CPU、编译、内核、输入哈希）
#   2. Release 构建
#   3. /usr/bin/time -v 运行 pipeline_performance_benchmark（3+ 轮）
#   4. 原始日志永久保存到 log_dir
#   5. 生成证据汇总摘要
#
# 用法：
#   bash scripts/run_performance_benchmark.sh [iterations] [output_dir]
#
# 输出：
#   log_dir/ — 原始日志（每轮独立子目录 + 单轮原始 JSON）
#   golden_output/ — benchmark JSON（汇总 + 每轮详细 + 验证后删除的 MP4）
#   log_dir/SUMMARY.txt — 文本摘要
#   log_dir/ENVIRONMENT.txt — 环境证据
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/performance_benchmark"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

ITERATIONS="${1:-3}"
OUTPUT_DIR="${2:-${PROJECT_DIR}/golden_output}"

mkdir -p "$RUN_LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== M05 Performance Benchmark Runner ==="
echo "Timestamp: $(date)"
echo "Project: $PROJECT_DIR"
echo "Iterations: $ITERATIONS"
echo "Output: $OUTPUT_DIR"
echo "Log: $RUN_LOG_DIR"

# ============================================================================
# 1. 环境证据采集
# ============================================================================

ENV_FILE="$RUN_LOG_DIR/ENVIRONMENT.txt"
{
    echo "=== M05 Performance Benchmark Environment ==="
    echo "Timestamp: $(date -u) UTC"
    echo "Hostname: $(hostname)"
    echo ""
    echo "--- CPU ---"
    grep "model name" /proc/cpuinfo | head -1 || echo "Unknown"
    echo "CPU cores: $(nproc)"
    echo ""
    echo "--- Memory ---"
    free -h || echo "free not available"
    echo ""
    echo "--- OS ---"
    uname -a
    echo ""
    echo "--- Compiler ---"
    g++ --version 2>/dev/null | head -1 || echo "g++ not found"
    echo ""
    echo "--- CMake ---"
    cmake --version 2>/dev/null | head -1 || echo "cmake not found"
    echo ""
    echo "--- Input Files SHA256 ---"
    sha256sum testdata/golden/face.jpg testdata/golden/audio.wav \
        models/wav2lip/wav2lip.param models/wav2lip/wav2lip.bin \
        models/shape_predictor_68_face_landmarks.dat 2>/dev/null || echo "sha256sum failed"
    echo ""
    echo "--- Build Type ---"
    echo "NDEBUG=$(cpp -dM /dev/null </dev/null 2>/dev/null | grep NDEBUG || echo 'not defined (Debug-like)')"
} > "$ENV_FILE"

cat "$ENV_FILE"

# ============================================================================
# 2. Release 构建
# ============================================================================

BUILD_DIR="${PROJECT_DIR}/build/release_bench"
echo ""
echo "=== Building Release ==="

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    2>&1 | tee "$RUN_LOG_DIR/cmake_configure.log"

cmake --build "$BUILD_DIR" --parallel "$(nproc)" \
    2>&1 | tee "$RUN_LOG_DIR/cmake_build.log"

BENCH_BIN="${BUILD_DIR}/bin/pipeline_performance_benchmark"
if [[ ! -f "$BENCH_BIN" ]]; then
    echo "ERROR: Benchmark binary not found at $BENCH_BIN"
    exit 1
fi

echo "Build completed: $BENCH_BIN"

# ============================================================================
# 3. 运行基准（/usr/bin/time -v）
# ============================================================================

echo ""
echo "=== Running Benchmark ($ITERATIONS iterations) ==="

cd "$PROJECT_DIR"

# 使用 /usr/bin/time -v 采集资源指标
# 第三个参数 raw_log_dir：单轮原始 JSON（含帧轨迹）写入 logs/performance_benchmark/<ts>/
BENCH_LOG="$RUN_LOG_DIR/benchmark_run.log"
TIME_LOG="$RUN_LOG_DIR/time_v_output.txt"

/usr/bin/time -v -o "$TIME_LOG" \
    "$BENCH_BIN" "$ITERATIONS" "$OUTPUT_DIR" "$RUN_LOG_DIR" \
    2>&1 | tee "$BENCH_LOG"

BENCH_EXIT_CODE="${PIPESTATUS[0]}"

echo ""
echo "Benchmark exit code: $BENCH_EXIT_CODE"

# ============================================================================
# 4. 收集结果
# ============================================================================

SUMMARY_FILE="$RUN_LOG_DIR/SUMMARY.txt"
{
    echo "=== M05 Performance Benchmark Summary ==="
    echo "Timestamp: $(date -u) UTC"
    echo "Iterations: $ITERATIONS"
    echo "Benchmark exit code: $BENCH_EXIT_CODE"
    echo ""
    echo "--- Resource Usage (/usr/bin/time -v) ---"
    grep -E "(User time|System time|Elapsed|Maximum resident|Minor|Major|Voluntary|Involuntary)" "$TIME_LOG" || echo "Resource info not available"
    echo ""
    echo "--- Benchmark JSON ---"
    if [[ -f "$OUTPUT_DIR/pipeline_performance_benchmark.json" ]]; then
        echo "Summary JSON: $OUTPUT_DIR/pipeline_performance_benchmark.json"
        echo "File size: $(stat --format=%s "$OUTPUT_DIR/pipeline_performance_benchmark.json" 2>/dev/null || echo "unknown") bytes"
    else
        echo "WARNING: Summary JSON not found"
    fi
    echo ""
    echo "--- Per-run JSON Files ---"
    ls -la "$OUTPUT_DIR"/pipeline_benchmark_run_*.json 2>/dev/null || echo "No per-run JSON files found"
    echo ""
    echo "--- Raw Per-run JSON Files (frame traces + ffprobe) ---"
    ls -la "$RUN_LOG_DIR"/pipeline_benchmark_run_raw_*.json 2>/dev/null || echo "No raw per-run JSON files found"
    echo ""
    echo "--- MP4 Verification ---"
    echo "Benchmark writes an MP4 per run via FinalMediaWriter, verifies it with ffprobe (streams, frame count, codec, size), then deletes it. Details in per-run JSONs."
    echo ""
    echo "--- Log Files ---"
    echo "Environment: $ENV_FILE"
    echo "Build log: $RUN_LOG_DIR/cmake_build.log"
    echo "Benchmark log: $BENCH_LOG"
    echo "Time -v output: $TIME_LOG"
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# ============================================================================
# 5. 复制 JSON 结果到日志目录（双重保存）
# ============================================================================

mkdir -p "$RUN_LOG_DIR/benchmark_output"
cp "$OUTPUT_DIR"/pipeline_performance_benchmark.json "$RUN_LOG_DIR/benchmark_output/" 2>/dev/null || echo "WARNING: Could not copy summary JSON"
cp "$OUTPUT_DIR"/pipeline_benchmark_run_*.json "$RUN_LOG_DIR/benchmark_output/" 2>/dev/null || echo "WARNING: Could not copy per-run JSONs"

echo ""
echo "=== BENCHMARK RUNNER COMPLETED ==="
echo "Log directory: $RUN_LOG_DIR"
echo "Summary: $SUMMARY_FILE"

exit $BENCH_EXIT_CODE
