#!/usr/bin/env bash
# =============================================================================
# M05 run_sanitizer_checks.sh — ASAN/TSAN 构建 + 测试验证
# =============================================================================
#
# 功能：
#   1. 构建 ASAN 版本，运行全部测试，检查 0 report
#   2. 构建 TSAN 版本，运行核心（pipeline/sync）测试，检查 0 report
#   3. 记录原始日志到 log_dir
#
# 用法：
#   bash scripts/run_sanitizer_checks.sh
#
# 输出：
#   log_dir/ — 原始日志（ASAN 和 TSAN 独立子目录）
#   log_dir/SANITIZER_SUMMARY.txt — 文本摘要
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/sanitizer_checks"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

mkdir -p "$RUN_LOG_DIR/asan"
mkdir -p "$RUN_LOG_DIR/tsan"

echo "=== M05 Sanitizer Checks ==="
echo "Timestamp: $(date)"
echo "Project: $PROJECT_DIR"
echo "Log: $RUN_LOG_DIR"

cd "$PROJECT_DIR"

# ============================================================================
# 1. ASAN 构建 + 全量测试
# ============================================================================

echo ""
echo "=== [1/2] ASAN Build + Full Test Suite ==="

ASAN_BUILD_DIR="${PROJECT_DIR}/build/asan_check"
rm -rf "$ASAN_BUILD_DIR"

cmake -B "$ASAN_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_ASAN=ON \
    -DENABLE_TSAN=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=OFF \
    2>&1 | tee "$RUN_LOG_DIR/asan/cmake_configure.log"

cmake --build "$ASAN_BUILD_DIR" --parallel "$(nproc)" \
    2>&1 | tee "$RUN_LOG_DIR/asan/cmake_build.log"

echo ""
echo "Running ASAN CTest..."
cd "$ASAN_BUILD_DIR"

# 检测 sanitizer 报告：任何 "ERROR:" 即失败
ASAN_LOG="$RUN_LOG_DIR/asan/ctest_output.log"
set +e
ctest --output-on-failure -j"$(nproc)" \
    2>&1 | tee "$ASAN_LOG"
ASAN_CTEST_EXIT="${PIPESTATUS[0]}"
set -e

# 检查 sanitizer 报告
ASAN_ERROR_COUNT=0
if grep -qE "(ERROR: AddressSanitizer|==[0-9]+==ERROR|SUMMARY: AddressSanitizer)" "$ASAN_LOG" 2>/dev/null; then
    ASAN_ERROR_COUNT=$(grep -cE "(ERROR: AddressSanitizer|==[0-9]+==ERROR|SUMMARY: AddressSanitizer)" "$ASAN_LOG" || echo 1)
fi

echo ""
echo "ASAN CTest exit code: $ASAN_CTEST_EXIT"
echo "ASAN error reports: $ASAN_ERROR_COUNT"

cd "$PROJECT_DIR"

# ============================================================================
# 2. TSAN 构建 + 核心测试（pipeline, sync）
# ============================================================================

echo ""
echo "=== [2/2] TSAN Build + Core Test Suite ==="

TSAN_BUILD_DIR="${PROJECT_DIR}/build/tsan_check"
rm -rf "$TSAN_BUILD_DIR"

cmake -B "$TSAN_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_ASAN=OFF \
    -DENABLE_TSAN=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=OFF \
    2>&1 | tee "$RUN_LOG_DIR/tsan/cmake_configure.log"

cmake --build "$TSAN_BUILD_DIR" --parallel "$(nproc)" \
    2>&1 | tee "$RUN_LOG_DIR/tsan/cmake_build.log"

echo ""
echo "Running TSAN CTest (pipeline and sync tests only)..."
cd "$TSAN_BUILD_DIR"

TSAN_LOG="$RUN_LOG_DIR/tsan/ctest_output.log"
set +e
# 只运行 pipeline 和 sync 相关测试（多线程路径）
ctest --output-on-failure -j1 \
    -R "(pipeline\.|bounded_task_queue|audio_video_synchronous|portaudio_playback|frame_scheduler|timestamp_manager)" \
    --timeout 600 \
    2>&1 | tee "$TSAN_LOG"
TSAN_CTEST_EXIT="${PIPESTATUS[0]}"
set -e

# 检查 TSAN 报告
TSAN_ERROR_COUNT=0
if grep -qE "(WARNING: ThreadSanitizer|SUMMARY: ThreadSanitizer|data race)" "$TSAN_LOG" 2>/dev/null; then
    TSAN_ERROR_COUNT=$(grep -cE "(WARNING: ThreadSanitizer|SUMMARY: ThreadSanitizer|data race)" "$TSAN_LOG" || echo 1)
fi

echo ""
echo "TSAN CTest exit code: $TSAN_CTEST_EXIT"
echo "TSAN error reports: $TSAN_ERROR_COUNT"

cd "$PROJECT_DIR"

# ============================================================================
# 3. 摘要
# ============================================================================

SUMMARY_FILE="$RUN_LOG_DIR/SANITIZER_SUMMARY.txt"
{
    echo "=== M05 Sanitizer Check Summary ==="
    echo "Timestamp: $(date -u) UTC"
    echo ""
    echo "--- ASAN ---"
    echo "CTest exit code: $ASAN_CTEST_EXIT"
    echo "AddressSanitizer errors: $ASAN_ERROR_COUNT"
    echo "Log: $ASAN_LOG"
    echo ""
    echo "--- TSAN ---"
    echo "CTest exit code: $TSAN_CTEST_EXIT"
    echo "ThreadSanitizer errors: $TSAN_ERROR_COUNT"
    echo "Log: $TSAN_LOG"
    echo ""
    if [[ "$ASAN_ERROR_COUNT" -eq 0 ]] && [[ "$TSAN_ERROR_COUNT" -eq 0 ]]; then
        echo "VERDICT: PASS — 0 sanitizer reports for both ASAN and TSAN."
    else
        echo "VERDICT: FAIL — Sanitizer reports detected."
        if [[ "$ASAN_ERROR_COUNT" -gt 0 ]]; then
            echo "  ASAN errors: $ASAN_ERROR_COUNT"
        fi
        if [[ "$TSAN_ERROR_COUNT" -gt 0 ]]; then
            echo "  TSAN errors: $TSAN_ERROR_COUNT"
        fi
    fi
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "=== SANITIZER CHECKS COMPLETED ==="
echo "Log directory: $RUN_LOG_DIR"
echo "Summary: $SUMMARY_FILE"

# 退出码：任何 sanitizer 错误 → 非零
if [[ "$ASAN_ERROR_COUNT" -eq 0 ]] && [[ "$TSAN_ERROR_COUNT" -eq 0 ]]; then
    exit 0
else
    exit 1
fi
