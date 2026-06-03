#!/bin/bash
# 监控消融实验，确保不会中断

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_FILE="monitor_ablations.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_processes() {
    local singlequery_pid=$(pgrep -f "veil_27b_singlequery.py" | head -1)
    local oracle_pid=$(pgrep -f "veil_27b_oracle.py" | head -1)
    local vllm_8778=$(pgrep -f "port 8778" | grep vllm | head -1)
    local vllm_8779=$(pgrep -f "port 8779" | grep vllm | head -1)

    local all_ok=1

    if [ -z "$singlequery_pid" ]; then
        log_msg "❌ ERROR: veil_27b_singlequery 进程已停止"
        all_ok=0
    else
        log_msg "✓ veil_27b_singlequery (PID $singlequery_pid) 运行中"
    fi

    if [ -z "$oracle_pid" ]; then
        log_msg "❌ ERROR: veil_27b_oracle 进程已停止"
        all_ok=0
    else
        log_msg "✓ veil_27b_oracle (PID $oracle_pid) 运行中"
    fi

    if [ -z "$vllm_8778" ]; then
        log_msg "❌ ERROR: vLLM 8778 进程已停止"
        all_ok=0
    else
        log_msg "✓ vLLM 8778 运行中"
    fi

    if [ -z "$vllm_8779" ]; then
        log_msg "❌ ERROR: vLLM 8779 进程已停止"
        all_ok=0
    else
        log_msg "✓ vLLM 8779 运行中"
    fi

    return $all_ok
}

check_results() {
    local oracle_lines=$(wc -l < "outputs/results/videommeL/veil_27b_oracle.jsonl" 2>/dev/null || echo 0)
    local singlequery_lines=$(wc -l < "outputs/results/videommeL/veil_27b_singlequery.jsonl" 2>/dev/null || echo 0)

    log_msg "📊 结果文件行数 - oracle: $oracle_lines, singlequery: $singlequery_lines"
}

check_gpu() {
    log_msg "🖥️ GPU 利用率:"
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk '{print "   GPU " $1 ": " $2}'
}

while true; do
    log_msg "--- 检查消融实验状态 ---"
    check_processes
    check_results
    check_gpu

    log_msg "--- 下次检查在 5 分钟后 ---"
    sleep 300
done
