# OmniFabric 在 RTX 3090（24 GB）上跑

> 💡 **一句話**：OmniFabric（SIGGRAPH Asia 2026）給一張衣服照片，就產出 GarmentCode 版片＋**版片 UV 上的貼圖**（去掉影子、皺褶，含印花）＋縫好的 3D 衣服。原版要 48 GB 顯示卡，這個資料夾把它改成 24 GB 能跑，並加上「用真的背面照」。
> 📅 2026-11-07
> ⚠️ **還沒在 3090 上實際跑過**：寫的時候手邊沒有 24 GB 顯示卡。改法照 diffusers 0.39 的程式碼確認過，但第一次跑很可能要修一兩個地方，log 帶回來給我看。

---

## 這個資料夾有什麼

| 檔案 | 做什麼 |
|---|---|
| `setup_3090.sh` | 一次裝好：下載 OmniFabric（固定版本）、建 conda 環境、安裝、放進下面的檔案、檢查權重 |
| `run_3090.sh` | 跑一件衣服：`正面照 [背面照] → 版片＋UV 貼圖＋3D` |
| `batch_3090.sh` | 跑一整個資料夾，跑完把要帶回來的檔案收進 `collect/` |
| `lowvram/flux2_lowvram.py` | FLUX.2 兩個步驟改成分兩段載入（24 GB 放得下） |
| `lowvram/flux1_lowvram.py` | FLUX.1 整理貼圖那一步改成分層搬運（24 GB 放得下） |

## 跟原版的差別

| | 原版（48 GB） | 這裡（24 GB） |
|---|---|---|
| 正面上色（FLUX.2，第 3 步） | 文字編碼器＋主模型同時在顯示卡，約 28 GB | 先跑文字編碼器、釋放，再載入主模型。**同樣的 4-bit 模型，結果應該相同**，多一次載入時間 |
| 背面（第 4 步） | 只有一張照片：MiniMax-H3 轉 360° 影片（要 ComfyUI），或 FLUX.2 推測背面 | **有背面照就用背面照上色**（同一個 FLUX.2 上色步驟，提示詞改成背面）；沒有才讓 FLUX.2 推測。不需要 ComfyUI |
| 整理貼圖（FLUX.1＋LoRA，第 6 步） | 全部放顯示卡，約 35 GB | 主模型權重放系統記憶體，算到哪層搬哪層。**精度不變**，比較慢 |
| 正面上色方式 | 可選 Gemini（手動）或 FLUX.2 | 固定 FLUX.2（全自動） |
| 做哪幾件 | 照片裡預測出的全部 | 預設只做上衣（`GARMENTS=upper`），要全部設 `GARMENTS=all` |

原版程式碼一行都沒改：兩個 `lowvram/*.py` 只是在執行時換掉它載入模型的函式，`run_3090.sh` 是原版 `scripts/run_pipeline.sh` 改了呼叫的模組。

---

## 一、事前準備（要你自己做的）

1. **電腦**：Linux、NVIDIA 24 GB 顯示卡、**系統記憶體建議 64 GB**（至少約 40 GB）、硬碟約 **200 GB** 空位（模型約 160 GB＋環境）
2. **conda**（Miniconda 就可以）
3. **CUDA 11.8 toolkit**（不是只有驅動程式；要有 `nvcc`）：https://developer.nvidia.com/cuda-11-8-0-download-archive
   可以和其他版本的 CUDA 並存，裝在 `/usr/local/cuda-11.8`
4. **Hugging Face 帳號**，到網站上**同意這兩個模型的授權**（要自己按，非商用授權，專題可以用）：
   - https://huggingface.co/black-forest-labs/FLUX.1-dev
   - https://huggingface.co/black-forest-labs/FLUX.2-dev

   然後在 3090 上執行 `huggingface-cli login` 貼上 token
5. **OmniFabric LoRA**（Google Drive，手動下載）：https://drive.google.com/file/d/1qMdaFTy8Mbh4yGcm41FE4iMOqUrGIp9k/view?usp=sharing
6. **ChatGarment 權重**：3090 上跑 `infer.py` 用的那份 `pytorch_model.bin`（14 GB）可以直接用，不用重新下載

## 二、安裝（一次）

把這個資料夾整個複製到 3090，然後：

```bash
export CHATGARMENT_BIN=<已經有的 ChatGarment pytorch_model.bin 路徑>
export HF_HOME=<有 160 GB 空位的資料夾>/huggingface      # 模型快取，之後每次跑也要設同一個
bash setup_3090.sh ~/OmniFabric
```

- 會裝在 `~/OmniFabric`，conda 環境叫 `omnifabric`
- 不編譯 flash-attn（原版要多花約 2 小時），ChatGarment 改用 PyTorch 內建的注意力，結果幾乎一樣
- 最後會列出還缺什麼。把 LoRA 放到 `~/OmniFabric/checkpoints/omnifabric/default.safetensors`

## 三、試跑一件

```bash
conda activate omnifabric
export HF_HOME=<同上>
cd ~/OmniFabric
bash lowvram/run_3090.sh examples/dress_1.png                           # 作者的範例（只有正面）
bash lowvram/run_3090.sh 照片/g07_front.jpg 照片/g07_back.jpg            # 我們的衣服（正面＋背面）
```

- **第一次跑會自動下載**：FLUX.2-dev 約 110 GB、FLUX.1-dev 約 34 GB、Hunyuan3D-2、Real-ESRGAN，要等很久
- 每一步會印出 `=== [n/6] …（已經過 x 分）`，最後印出總時間
- 時間我沒辦法實測。粗估一件 20～40 分鐘（不含第一次下載），最慢的是第 6 步

## 四、整批跑

照片放同一個資料夾，命名 `<衣服>_front.jpg`、`<衣服>_back.jpg`（背面可以沒有）：

```bash
bash lowvram/batch_3090.sh ~/衣服照片 outputs/batch
```

一件失敗不會中斷，最後列出失敗的；每件的 log 在 `outputs/batch/<衣服>.log`。

**建議先跑第二批**：`output/chatgarment_ai/inputs_h/`（正面）和 `inputs_h_back/`（背面）是同一件衣服的正背面官方照，
最適合測「用真的背面照」。這兩個資料夾在跑過 ChatGarment 的那台電腦上（不在 GitHub），複製到 3090 後改名成
`h05_frenchblue_button_down_shirt_front.jpg`、`h05_frenchblue_button_down_shirt_back.jpg` 這種格式。

## 五、帶回來

只要帶 `outputs/batch/collect/`，每件：

| 檔案 | 是什麼 |
|---|---|
| `normalized_texture.png` | **最終 UV 貼圖**（版片空間） |
| `editable_template.png` | 版片 UV 範本，看哪一塊是哪片版片 |
| `front_garment.png`、`back_garment.png` | 正、背面上色結果，檢查用 |
| `final/` | 貼好貼圖、縫好的 3D 衣服（obj、mtl、渲染圖） |
| `design/` | ChatGarment 預測的款式 |

放到 `D:\newfasion_VT_work\repo_deps\output\omnifabric\`。商品照做出來的貼圖**不要放進 GitHub**（和照片同樣的版權問題）。

---

## 六、出問題時

| 現象 | 試試看 |
|---|---|
| 第 3 或第 4 步印出 `第 2 段：FLUX.2 主模型` 之後 `CUDA out of memory` | `export FLUX2_MAX_SIDE=768` 再跑（輸出解析度降低） |
| 第 3 或第 4 步在 `第 1 段` 就爆 | 文字編碼器本身放不下，把 log 帶回來 |
| 第 6 步系統記憶體不夠（被系統砍掉、`Killed`） | `export LOWVRAM_NO_STREAM=1` 再跑（更慢但更省記憶體） |
| 第 6 步很慢 | 正常，主模型每一步都要從系統記憶體搬上顯示卡 |
| `check_checkpoints` 說缺 FLUX | 還沒同意授權或沒登入（見「一、事前準備」第 4 點） |
| 安裝時 `Set CUDA_HOME to a CUDA 11.8 toolkit` | 沒裝 CUDA 11.8 toolkit，或 `CUDA_HOME` 沒指對 |
| 其他錯誤 | 重跑同一個指令會從做完的地方接著做；不行就把 `<衣服>.log` 最後 50 行帶回來 |

## 七、授權

- OmniFabric 程式碼：Apache 2.0（`run_3090.sh` 改自它的 `scripts/run_pipeline.sh`）
- FLUX.1-dev、FLUX.2-dev：**非商用**授權
- Hunyuan3D-2：Tencent Hunyuan 3D 社群授權
- ChatGarment 權重含 Llama 2：LLAMA 2 Community License
- 這裡沒有用到 MiniMax-H3（它的授權不涵蓋美國、英國、韓國、歐盟）

## 參考

- OmniFabric：https://github.com/humansensinglab/OmniFabric（專案頁 https://humansensinglab.github.io/OmniFabric/）
- 論文：https://arxiv.org/abs/2609.30234
