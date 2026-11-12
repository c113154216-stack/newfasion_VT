#!/bin/bash
# 停掉 go_bg.sh 開的背景工作（連同它開的所有子程式）。之後再跑 go_bg.sh 會從做完的地方接著做。
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
OUT=$(eval echo "$OUT")
PIDFILE="$OUT/go.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill -- -"$(cat "$PIDFILE")" && echo "已停掉（PID $(cat "$PIDFILE")）"
    rm -f "$PIDFILE"
else
    echo "沒有在跑。"
fi
