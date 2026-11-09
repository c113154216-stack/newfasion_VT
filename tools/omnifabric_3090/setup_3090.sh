#!/bin/bash
# 在 3090 那台（Linux）裝好 OmniFabric，並放進我們的 24 GB 版本。只要跑一次。
#
#   bash setup_3090.sh [安裝位置，預設 ~/OmniFabric]
#
# 事前要準備（見 README「一、事前準備」）：conda、CUDA 11.8 toolkit、Hugging Face 帳號並同意 FLUX.1-dev／FLUX.2-dev 授權。
# 環境變數：
#   CUDA_HOME        CUDA 11.8 toolkit 位置（預設 /usr/local/cuda-11.8）
#   CHATGARMENT_BIN  已經有的 ChatGarment 權重 pytorch_model.bin（3090 上跑 infer.py 用的那份，14 GB），有給就直接連過去不用再下載
#   HF_HOME          Hugging Face 模型快取位置，要有約 160 GB 空間

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=$(realpath -m "${1:-$HOME/OmniFabric}")
COMMIT=205c17947e71bc57af2823528271617ea04ef389   # 2026-11-02，這份腳本是照這一版改的
# CUDA 11.8 toolkit：有 /usr/local/cuda-11.8 就用；沒有就在 conda 環境裡裝一份（不用 sudo）
[[ -z "$CUDA_HOME" && -x /usr/local/cuda-11.8/bin/nvcc ]] && export CUDA_HOME=/usr/local/cuda-11.8

echo "== 1. 下載 OmniFabric（固定版本 ${COMMIT:0:7}）到 $DEST"
if [[ ! -d "$DEST" ]]; then
    git clone https://github.com/humansensinglab/OmniFabric.git "$DEST"
fi
git -C "$DEST" checkout "$COMMIT"

echo "== 2. 建 conda 環境並安裝（不編譯 flash-attn，省 2 小時；ChatGarment 改用 PyTorch 內建的注意力）"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^omnifabric " || conda env create -f "$DEST/environment.yml"
conda activate omnifabric
if [[ -z "$CUDA_HOME" ]]; then
    [[ -x "$CONDA_PREFIX/bin/nvcc" ]] || conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
    export CUDA_HOME=$CONDA_PREFIX
fi
if [[ ! -f "$DEST/.installed_3090" ]]; then   # 裝過就跳過（重跑 setup 只會補權重、更新 lowvram/）
    (cd "$DEST" && bash install.sh --no-flash-attn)
    touch "$DEST/.installed_3090"
fi

echo "== 3. 放進 24 GB 版本（lowvram/）"
mkdir -p "$DEST/lowvram"
cp "$HERE/lowvram/"*.py "$HERE/run_3090.sh" "$HERE/batch_3090.sh" "$DEST/lowvram/"

echo "== 4. 權重"
mkdir -p "$DEST/checkpoints/chatgarment" "$DEST/checkpoints/omnifabric"
if [[ -z "$CHATGARMENT_BIN" ]]; then   # 沒指定就在家目錄找 3090 上已經有的那份（大於 10 GB 的 pytorch_model.bin）
    CHATGARMENT_BIN=$(find "$HOME" -maxdepth 6 -name pytorch_model.bin -size +10G 2>/dev/null | grep -i chatgarment | head -n 1)
    [[ -n "$CHATGARMENT_BIN" ]] && echo "  找到 ChatGarment 權重：$CHATGARMENT_BIN"
fi
if [[ -n "$CHATGARMENT_BIN" && -f "$CHATGARMENT_BIN" && ! -e "$DEST/checkpoints/chatgarment/pytorch_model.bin" ]]; then
    ln -s "$(realpath "$CHATGARMENT_BIN")" "$DEST/checkpoints/chatgarment/pytorch_model.bin"
fi
missing=0
if [[ ! -e "$DEST/checkpoints/chatgarment/pytorch_model.bin" ]]; then
    echo "  缺 ChatGarment 權重：設 CHATGARMENT_BIN 指到已經有的 pytorch_model.bin 再跑一次，或從"
    echo "    https://drive.google.com/drive/folders/17JVZ1HYgMMMxw_zWmDgiVQ_PkIN-Fznf"
    echo "  下載成 $DEST/checkpoints/chatgarment/pytorch_model.bin"
    missing=1
fi
if [[ ! -f "$DEST/checkpoints/omnifabric/default.safetensors" ]]; then   # Google Drive 自動下載
    pip install -q gdown && gdown 1qMdaFTy8Mbh4yGcm41FE4iMOqUrGIp9k -O "$DEST/checkpoints/omnifabric/default.safetensors" || true
fi
if [[ ! -f "$DEST/checkpoints/omnifabric/default.safetensors" ]]; then
    echo "  缺 OmniFabric LoRA（自動下載失敗）：從"
    echo "    https://drive.google.com/file/d/1qMdaFTy8Mbh4yGcm41FE4iMOqUrGIp9k/view?usp=sharing"
    echo "  下載成 $DEST/checkpoints/omnifabric/default.safetensors"
    missing=1
fi
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "  還沒登入 Hugging Face：先到網站同意 FLUX.1-dev、FLUX.2-dev 的授權，再執行 huggingface-cli login"
    missing=1
fi

echo ""
[[ $missing == 0 ]] || { echo "上面列出的東西補齊後，再跑一次同一個指令。"; exit 2; }
if [[ $missing == 0 ]]; then
    echo "安裝完成。試跑一件："
else
    echo "安裝完成，但上面列出的東西要補齊才能跑。補齊後："
fi
echo "  conda activate omnifabric && cd $DEST"
echo "  bash lowvram/run_3090.sh <正面照> [背面照]"
echo "第一次跑會自動下載 FLUX.2-dev（約 110 GB）、FLUX.1-dev（約 34 GB）、Hunyuan3D-2、LLaVA 設定檔、Real-ESRGAN。"
