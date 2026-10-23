# ChatGarment 推論結果

AI 判斷款式的原始輸出。重新產生要用顯示卡（bf16 要 24 GB 的 RTX 3090），所以保存在 repo 裡；
網格、對照圖可以隨時用 `to_design.py` 重做，放在 `output/`（不上傳）。

| 資料夾 | 模式 | 機器 | 件數 |
|---|---|---|---|
| `4bit_v2/` | 4-bit（NF4） | RTX 5050，Windows | 8 |
| `bf16_3090/` | bf16 原版 | RTX 3090，Linux | 6（黑 T、魚印花 T 是 webp，沒跑到） |
| `bf16_h/` | bf16 原版 | RTX 3090，Linux | 12（第二批：短版、長版、V 領、Polo、襯衫、背心、拉鍊帽 T、開襟毛衣、高領、落肩、條紋、背面印花） |
| `4bit_h/` | 4-bit（NF4） | RTX 5050，Windows | 12（同上） |

照片在 `output/chatgarment_ai/inputs_v2/`（第一批）、`inputs_h/`＋`inputs_h_back/`（第二批，正面＋背面），由 `tools/chatgarment/import_photos.py` 改名而來。
第二批是品牌商品頁的官方照，**只供內部測試，不上傳、不公開**（來源見 `inputs_h/sources.md`）。
`bf16_3090/` 在 3090 上是用原始檔名跑的（`Hoodie_1`、`RedT`…），帶回後改成相同的 g 開頭名稱。

每張照片一個資料夾：

| 檔案 | 內容 |
|---|---|
| `output.txt` | 模型的兩段文字輸出：[描述]（第一問）、[版型]（第二問） |
| `output.json` | 修復後的版型 JSON |
| `float_preds.npy` | 76 個連續數值（衣長、寬鬆度、袖長…） |
| `meta.json` | 耗時、顯示記憶體峰值 |
| `design_upper.yaml`、`design_lower.yaml` | `to_design.py` 轉出的 GarmentCode 款式檔（已套用領子修正） |

比較報告：`report_4bit_v2_vs_bf16_3090.md`（`compare.py` 產生）。

重做網格：
```
.venv\Scripts\python tools\chatgarment\to_design.py results\chatgarment_ai\bf16_3090 --tag aibf16_3090
```
