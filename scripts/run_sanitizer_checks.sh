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
#
# PASS 判定（ASAN 与 TSAN 均须满足，缺一即 FAIL）：
#   - CTest 退出码为 0（失败/超时/崩溃/未启动 → 非零 → FAIL）
#   - 日志中无 sanitizer 错误报告（ASAN/TSAN ERROR、WARNING、SUMMARY 等）
#   - 所有强制测试实际运行：日志中无 ***Not Run / ***Skipped /
#     "The following tests did not run" / "No tests were found"，且
#     期望测试（全量套件 / TSAN 正则匹配集）全部出现在运行结果中
#   - TSAN 必须以 setarch x86_64 -R 运行（禁用 ASLR，TSAN 必需）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/sanitizer_checks"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# TSAN 核心（多线程路径）测试正则
TSAN_TEST_REGEX="(pipeline\.|bounded_task_queue|audio_video_synchronous|portaudio_playback|frame_scheduler|timestamp_manager)"

mkdir -p "$RUN_LOG_DIR/asan"
mkdir -p "$RUN_LOG_DIR/tsan"

echo "=== M05 Sanitizer Checks ==="
echo "Timestamp: $(date)"
echo "Project: $PROJECT_DIR"
echo "Log: $RUN_LOG_DIR"

cd "$PROJECT_DIR"

# ============================================================================
# 辅助函数
# ============================================================================

# 检测 CTest 日志中是否存在 NOT_RUN / SKIPPED / 无测试匹配
# 注意：ctest -R 无匹配（"No tests were found!!!"）与跳过（***Skipped）时
# 退出码仍为 0，必须解析日志才能识别"测试未运行"。
ctest_had_not_run_or_skipped() {
    local log="$1"
    grep -qE "(\*\*\*Not Run|\*\*\*Skipped|No tests were found|The following tests did not run)" "$log" 2>/dev/null
}

# 验证每个期望测试名都出现在 ctest 运行日志中（即确实被调度执行）
# 用法: check_expected_tests_ran "LOG" <name> [<name> ...]
# 缺失时打印缺失列表并返回 1
check_expected_tests_ran() {
    local log="$1"
    shift
    local missing=0
    local t escaped
    for t in "$@"; do
        escaped="$(printf '%s' "$t" | sed 's/[.]/\\&/g')"
        if ! grep -qE "Test *#[0-9]+: ${escaped}(\\.| )" "$log"; then
            echo "    [FAIL] test did not run / absent from results: $t"
            missing=1
        fi
    done
    return "$missing"
}

# 从 ctest -N 输出中提取期望测试名列表（多行文本）
extract_expected_test_names() {
    sed -n 's/^ *Test *#[0-9]*: //p' "$1"
}

# 统计非空期望测试数（注意：grep -c 在计数为 0 时返回非零，
# 在 set -euo pipefail 下会中止脚本，因此用始终返回 0 的 wc -l）
count_nonempty_lines() {
    printf '%s\n' "$1" | sed 's/^[[:space:]]*//; /^$/d' | wc -l
}

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
echo "Running ASAN CTest (full suite)..."
cd "$ASAN_BUILD_DIR"

# 枚举全量期望测试（ASAN 运行全部注册测试，缺一即 FAIL）
ASAN_EXPECTED_LOG="$RUN_LOG_DIR/asan/expected_tests.txt"
ctest -N > "$ASAN_EXPECTED_LOG" 2>&1
ASAN_EXPECTED_TESTS="$(extract_expected_test_names "$ASAN_EXPECTED_LOG")"
ASAN_EXPECTED_COUNT="$(count_nonempty_lines "$ASAN_EXPECTED_TESTS")"

# 检测 sanitizer 报告：任何 "ERROR:" 即失败
ASAN_LOG="$RUN_LOG_DIR/asan/ctest_output.log"
set +e
ctest --output-on-failure -j"$(nproc)" \
    --timeout 900 \
    2>&1 | tee "$ASAN_LOG"
ASAN_CTEST_EXIT="${PIPESTATUS[0]}"
set -e

# 检查 sanitizer 报告
ASAN_ERROR_COUNT=0
if grep -qE "(ERROR: AddressSanitizer|==[0-9]+==ERROR|SUMMARY: AddressSanitizer)" "$ASAN_LOG" 2>/dev/null; then
    ASAN_ERROR_COUNT=$(grep -cE "(ERROR: AddressSanitizer|==[0-9]+==ERROR|SUMMARY: AddressSanitizer)" "$ASAN_LOG" || echo 1)
fi

# 检查 NOT_RUN / SKIPPED / 无测试匹配
ASAN_NOT_RUN="no"
if ctest_had_not_run_or_skipped "$ASAN_LOG"; then
    ASAN_NOT_RUN="yes"
fi

# 检查所有注册测试确实出现在运行结果中
ASAN_MISSING_REPORT="$(check_expected_tests_ran "$ASAN_LOG" $ASAN_EXPECTED_TESTS || true)"
ASAN_MISSING_TESTS="no"
if [[ -n "$ASAN_MISSING_REPORT" ]] || [[ "$ASAN_EXPECTED_COUNT" -eq 0 ]]; then
    ASAN_MISSING_TESTS="yes"
fi

echo ""
echo "ASAN CTest exit code: $ASAN_CTEST_EXIT"
echo "ASAN error reports: $ASAN_ERROR_COUNT"
echo "ASAN tests not run/skipped: $ASAN_NOT_RUN"
echo "ASAN expected tests registered: $ASAN_EXPECTED_COUNT"

# ASAN 判定：CTest exit=0 且 0 sanitizer 报告 且 全部测试实际运行
ASAN_OK=1
if [[ "$ASAN_CTEST_EXIT" -ne 0 ]]; then ASAN_OK=0; fi
if [[ "$ASAN_ERROR_COUNT" -ne 0 ]]; then ASAN_OK=0; fi
if [[ "$ASAN_NOT_RUN" == "yes" ]]; then ASAN_OK=0; fi
if [[ "$ASAN_MISSING_TESTS" == "yes" ]]; then ASAN_OK=0; fi

cd "$PROJECT_DIR"

# ============================================================================
# 2. TSAN 构建 + 核心测试（pipeline, sync）
# ============================================================================

echo ""
echo "=== [2/2] TSAN Build + Core Test Suite (pipeline/sync) ==="

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

# 枚举 TSAN 正则匹配的期望测试（强制测试集）
TSAN_EXPECTED_LOG="$RUN_LOG_DIR/tsan/expected_tests.txt"
ctest -N -R "$TSAN_TEST_REGEX" > "$TSAN_EXPECTED_LOG" 2>&1
TSAN_EXPECTED_TESTS="$(extract_expected_test_names "$TSAN_EXPECTED_LOG")"
TSAN_EXPECTED_COUNT="$(count_nonempty_lines "$TSAN_EXPECTED_TESTS")"

TSAN_LOG="$RUN_LOG_DIR/tsan/ctest_output.log"
if command -v setarch >/dev/null 2>&1; then
    # TSAN 必需：禁用地址空间布局随机化（ASLR），避免随机映射干扰 shadow memory
    echo "Running TSAN CTest with ASLR disabled (setarch x86_64 -R) as required..."
    set +e
    setarch x86_64 -R ctest --output-on-failure -j1 \
        -R "$TSAN_TEST_REGEX" \
        --timeout 600 \
        2>&1 | tee "$TSAN_LOG"
    TSAN_CTEST_EXIT="${PIPESTATUS[0]}"
    set -e
else
    echo "ERROR: setarch(1) not found — TSAN cannot be run with ASLR disabled (required). Treating as FAILURE." >&2
    TSAN_CTEST_EXIT=127
fi

# 检查 TSAN 报告
TSAN_ERROR_COUNT=0
if grep -qE "(WARNING: ThreadSanitizer|SUMMARY: ThreadSanitizer|data race)" "$TSAN_LOG" 2>/dev/null; then
    TSAN_ERROR_COUNT=$(grep -cE "(WARNING: ThreadSanitizer|SUMMARY: ThreadSanitizer|data race)" "$TSAN_LOG" || echo 1)
fi

# 检查 NOT_RUN / SKIPPED / 无测试匹配
TSAN_NOT_RUN="no"
if ctest_had_not_run_or_skipped "$TSAN_LOG"; then
    TSAN_NOT_RUN="yes"
fi

# 检查期望（正则匹配）测试全部出现在运行结果中
TSAN_MISSING_REPORT="$(check_expected_tests_ran "$TSAN_LOG" $TSAN_EXPECTED_TESTS || true)"
TSAN_MISSING_TESTS="no"
if [[ -n "$TSAN_MISSING_REPORT" ]] || [[ "$TSAN_EXPECTED_COUNT" -eq 0 ]]; then
    TSAN_MISSING_TESTS="yes"
fi

echo ""
echo "TSAN CTest exit code: $TSAN_CTEST_EXIT"
echo "TSAN error reports: $TSAN_ERROR_COUNT"
echo "TSAN tests not run/skipped: $TSAN_NOT_RUN"
echo "TSAN expected tests (regex match): $TSAN_EXPECTED_COUNT"

# TSAN 判定：CTest exit=0 且 0 sanitizer 报告 且 全部强制测试实际运行
TSAN_OK=1
if [[ "$TSAN_CTEST_EXIT" -ne 0 ]]; then TSAN_OK=0; fi
if [[ "$TSAN_ERROR_COUNT" -ne 0 ]]; then TSAN_OK=0; fi
if [[ "$TSAN_NOT_RUN" == "yes" ]]; then TSAN_OK=0; fi
if [[ "$TSAN_MISSING_TESTS" == "yes" ]]; then TSAN_OK=0; fi

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
    echo "Tests not run / skipped: $ASAN_NOT_RUN"
    echo "Expected tests registered: $ASAN_EXPECTED_COUNT"
    echo "Log: $ASAN_LOG"
    echo ""
    echo "--- TSAN ---"
    echo "CTest exit code: $TSAN_CTEST_EXIT"
    echo "ThreadSanitizer errors: $TSAN_ERROR_COUNT"
    echo "Tests not run / skipped: $TSAN_NOT_RUN"
    echo "Expected tests (regex match): $TSAN_EXPECTED_COUNT"
    echo "Log: $TSAN_LOG"
    echo ""
    if [[ "$ASAN_OK" -eq 1 ]] && [[ "$TSAN_OK" -eq 1 ]]; then
        echo "VERDICT: PASS — CTest exit=0, 0 sanitizer reports, all mandatory tests ran, TSAN ran with setarch x86_64 -R."
    else
        echo "VERDICT: FAIL"
        if [[ "$ASAN_OK" -ne 1 ]]; then
            echo "  ASAN failure reasons:"
            if [[ "$ASAN_CTEST_EXIT" -ne 0 ]]; then
                echo "    - CTest exit code $ASAN_CTEST_EXIT (non-zero)"
            fi
            if [[ "$ASAN_ERROR_COUNT" -gt 0 ]]; then
                echo "    - $ASAN_ERROR_COUNT AddressSanitizer report(s)"
            fi
            if [[ "$ASAN_NOT_RUN" == "yes" ]]; then
                echo "    - tests not run / skipped / none found in log"
            fi
            if [[ "$ASAN_MISSING_TESTS" == "yes" ]]; then
                echo "    - not all registered tests appeared in run results"
            fi
            if [[ -n "$ASAN_MISSING_REPORT" ]]; then
                echo "$ASAN_MISSING_REPORT"
            fi
        fi
        if [[ "$TSAN_OK" -ne 1 ]]; then
            echo "  TSAN failure reasons:"
            if [[ "$TSAN_CTEST_EXIT" -ne 0 ]]; then
                echo "    - CTest exit code $TSAN_CTEST_EXIT (non-zero)"
            fi
            if [[ "$TSAN_ERROR_COUNT" -gt 0 ]]; then
                echo "    - $TSAN_ERROR_COUNT ThreadSanitizer report(s)"
            fi
            if [[ "$TSAN_NOT_RUN" == "yes" ]]; then
                echo "    - tests not run / skipped / none found in log"
            fi
            if [[ "$TSAN_MISSING_TESTS" == "yes" ]]; then
                echo "    - not all expected (regex-matched) tests appeared in run results"
            fi
            if [[ -n "$TSAN_MISSING_REPORT" ]]; then
                echo "$TSAN_MISSING_REPORT"
            fi
        fi
    fi
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "=== SANITIZER CHECKS COMPLETED ==="
echo "Log directory: $RUN_LOG_DIR"
echo "Summary: $SUMMARY_FILE"

# 退出码：PASS 仅当 CTest exit=0 且无 sanitizer 报告且所有强制测试实际运行
if [[ "$ASAN_OK" -eq 1 ]] && [[ "$TSAN_OK" -eq 1 ]]; then
    exit 0
else
    exit 1
fi
