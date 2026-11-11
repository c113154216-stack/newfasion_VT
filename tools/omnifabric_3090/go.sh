#!/bin/bash
# 一個指令跑完 OmniFabric（3090）：沒裝就先裝，缺權重就補或告訴你缺什麼，然後整批跑。
#
#   bash go.sh          跑 PHOTOS 裡全部的衣服
#   bash go.sh h05      只跑名稱開頭是 h05 的（第一次建議先跑一件）
#
# 設定在 config.sh。中斷了直接重跑同一個指令，會從做完的地方接著做。
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
x() { eval echo "$1"; }   # 展開 ~
OMNI_DIR=$(x "$OMNI_DIR"); PHOTOS=$(x "$PHOTOS"); OUT=$(x "$OUT")
export HF_HOME=$(x "$HF_HOME") CHATGARMENT_BIN=$(x "$CHATGARMENT_BIN") GARMENTS
mkdir -p "$HF_HOME" "$OUT"
# huggingface-cli login 預設把 token 存在 ~/.cache/huggingface/token；改了 HF_HOME 就要複製過去
[ -f "$HF_HOME/token" ] || [ ! -f ~/.cache/huggingface/token ] || cp ~/.cache/huggingface/token "$HF_HOME/token"

[[ -d "$PHOTOS" ]] || { echo "找不到照片資料夾 $PHOTOS（改 config.sh 的 PHOTOS）"; exit 1; }
command -v conda > /dev/null || { echo "沒有 conda：先裝 Miniconda https://docs.anaconda.com/miniconda/"; exit 1; }

echo "########## 1. 安裝／檢查（裝過的會跳過）"
bash "$HERE/setup_3090.sh" "$OMNI_DIR" 2>&1 | tee "$OUT/setup.log"
[[ ${PIPESTATUS[0]} == 0 ]] || exit 1

echo "########## 2. 開始跑（每件約 20～40 分，log 在 $OUT/<衣服>.log）"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omnifabric
cd "$OMNI_DIR"
bash lowvram/batch_3090.sh "$PHOTOS" "$OUT" "$1"
echo "把 $OUT/collect/ 整個帶回來。"
