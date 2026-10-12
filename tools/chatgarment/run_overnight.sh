#!/bin/bash
# 整批跑 ChatGarment：4-bit 推論 → 轉 design.yaml＋生成網格＋縫合 → bf16 原版推論 → 轉 design.yaml → 比較報告
# 在 newfashion_VT 根目錄用 Git Bash 執行：bash tools/chatgarment/run_overnight.sh
# 每一步都會跳過已經完成的照片，中斷後重跑會從斷掉的地方接著做。
set -u
cd "$(dirname "$0")/../.."
ROOT=output/chatgarment_ai
LOG=$ROOT/run.log
AI=.venv-chatgarment/Scripts/python
GC=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
mkdir -p $ROOT

step() { echo; echo "===== $(date '+%F %T') $* =====" | tee -a $LOG; }
run() { "$@" 2>&1 | grep -v "^W[0-9]\{4\}\|Xet Storage\|instantiate a model of type" | tee -a $LOG; }

step "1. 4-bit 推論"
run $AI tools/chatgarment/infer.py --mode 4bit --images $ROOT/inputs --out $ROOT/4bit

step "2. 4-bit 結果 → design.yaml → 網格 → 縫合"
run $GC tools/chatgarment/to_design.py $ROOT/4bit --tag ai4bit

step "3. bf16（原版精度）推論"
# bf16 會跑好幾個小時，要用 CONFIRM_BF16=1 bash tools/chatgarment/run_overnight.sh 明確開啟
run $AI tools/chatgarment/infer.py --mode bf16 --images $ROOT/inputs --out $ROOT/bf16 ${CONFIRM_BF16:+--confirm-bf16}

step "4. bf16 結果 → design.yaml → 網格 → 縫合"
run $GC tools/chatgarment/to_design.py $ROOT/bf16 --tag aibf16

step "5. 比較報告"
run $GC tools/chatgarment/compare.py $ROOT

step "完成"
