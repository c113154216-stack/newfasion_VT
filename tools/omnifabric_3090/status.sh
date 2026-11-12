#!/bin/bash
# 看 go_bg.sh 的進度。   bash status.sh      bash status.sh -w（每 30 秒更新，Ctrl+C 離開，不會停掉程式）
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
OUT=$(eval echo "$OUT")
PIDFILE="$OUT/go.pid"
LOG="$OUT/go.log"

# 最後一行（把進度條的 \r 換成換行，去掉空行）
last_line() { tr '\r' '\n' < "$1" | grep -v '^[[:space:]]*$' | tail -n "${2:-1}"; }
mins_ago() { echo $(( ($(date +%s) - $(stat -c %Y "$1")) / 60 )); }

show() {
    echo "======== OmniFabric 進度  $(date '+%m-%d %H:%M')"
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "狀態：跑中（PID $(cat "$PIDFILE")）"
    elif [[ -f "$LOG" ]]; then
        echo "狀態：沒有在跑（已結束或被停掉）"
    else
        echo "狀態：還沒用 go_bg.sh 跑過"; return
    fi
    grep -q "等顯示卡空出來" "$LOG" && ! grep -q "開始$" "$LOG" && echo "  $(last_line "$LOG")"

    echo ""
    echo "-- 每件衣服"
    grep '^######## ' "$LOG" | sed 's/^######## //; s/（.*//' | while read -r NAME; do
        L="$OUT/$NAME.log"
        [[ -f "$L" ]] || continue
        if grep -q '^完成：' "$L"; then
            echo "  ✅ $NAME  $(grep '^完成：' "$L" | tail -n 1 | grep -o '共 [0-9]* 分')"
        elif grep -A1 "^######## $NAME" "$LOG" | grep -q '失敗'; then
            echo "  ❌ $NAME  失敗：$(last_line "$L" | cut -c1-120)"
        else
            STEP=$(grep '^=== \[[0-9]/6\]' "$L" | tail -n 1 | sed 's/^=== //; s/（已經過.*//')
            echo "  ▶ $NAME  ${STEP:-準備中}（log $(mins_ago "$L") 分鐘前更新）"
            echo "     $(last_line "$L" | cut -c1-120)"
        fi
    done

    echo ""
    echo "-- 顯示卡"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
        | awk -F', ' '{print "  使用率 " $1 "，記憶體 " $2 " / " $3}'
    echo ""
    echo "-- 做好的：$OUT/collect/"
    ls "$OUT/collect" 2>/dev/null | sed 's/^/  /'
    grep -q '全部結束' "$LOG" 2>/dev/null && echo "" && echo "全部跑完了，把 $OUT/collect/ 帶回來。"
}

if [[ "$1" == "-w" ]]; then
    while :; do clear; show; sleep 30; done
else
    show
fi
