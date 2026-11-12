#!/bin/bash
# 在背景跑 go.sh：關掉終端機、斷線都會繼續跑。顯示卡被別人佔著時會先等它空出來再開始。
#
#   bash go_bg.sh          背景跑 PHOTOS 裡全部的衣服
#   bash go_bg.sh h05      只跑名稱開頭是 h05 的
#
#   bash status.sh         看進度（加 -w 每 30 秒自動更新）
#   bash stop.sh           停掉
#
# 至少要空出多少顯示卡記憶體才開始（MiB），預設 20000：  MIN_FREE_MB=18000 bash go_bg.sh
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
OUT=$(eval echo "$OUT")
PIDFILE="$OUT/go.pid"
LOG="$OUT/go.log"
MIN_FREE_MB=${MIN_FREE_MB:-20000}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '; }

if [[ "$1" == "--inner" ]]; then
    shift
    while :; do
        free=$(gpu_free)
        [[ -n "$free" && "$free" -ge "$MIN_FREE_MB" ]] && break
        echo "[$(date '+%m-%d %H:%M')] 等顯示卡空出來：目前剩 ${free:-?} MiB，要 $MIN_FREE_MB MiB"
        sleep 60
    done
    echo "[$(date '+%m-%d %H:%M')] 顯示卡剩 $free MiB，開始"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    bash "$HERE/go.sh" "$@"
    echo "[$(date '+%m-%d %H:%M')] 全部結束"
    exit 0
fi

mkdir -p "$OUT"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "已經在跑了（PID $(cat "$PIDFILE")）。看進度：bash $HERE/status.sh"
    exit 1
fi
if pgrep -u "$USER" -f "lowvram/run_3090.sh" > /dev/null; then
    echo "有另一個 OmniFabric 正在跑（可能在 tmux 裡）。先把它停掉，或等它跑完。"
    exit 1
fi

# setsid：脫離這個終端機，斷線也不會被關掉
setsid nohup bash "$HERE/go_bg.sh" --inner "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "已在背景開始（PID $!），現在可以關掉終端機。"
echo "看進度：bash $HERE/status.sh      停掉：bash $HERE/stop.sh"
