# body_fit：照身體生成貼身衣物

計畫書 2.4 節的主流程：衣服依使用者身體＋版型餘量生成，不代表任何成衣號碼；號碼另外用尺寸表判定（`web/js/sizing.js`）。
AI（ChatGarment）只決定款式，大小交給身體。

完整步驟、每一步用哪支程式、目前結果：[`docs/流程_照片到網格.md`](../../docs/流程_照片到網格.md)

## 主要程式

| 步驟 | 程式 | 環境 |
|---|---|---|
| ① 整理照片 | `import_photos.py` | `.venv` |
| ② AI 推論 | `infer.py` | `.venv-chatgarment` |
| ③ 轉版型參數＋描述修正 | `to_design.py` | `.venv` |
| ④ 校正：領深、衣長、袖長 | `neckline.py`、`hemline.py`、`sleevelen.py` | `.venv` |
| ⑤ 打版 | `design_to_mesh.py` | `.venv` |
| ⑥ 縫合＋尖角攤平 | `sew.py` | `.venv` |
| ⑦ 匯出給 engine（④～⑦ 一次做完） | `export_neutral.py` | `.venv` |
| ⑧ 交付檢查 | `experiments/delivery_check.py`、`experiments/neutral_preview.py` | `.venv` |

其他：`chat_server.py`＋`chat_worker.py`＋`chat_web/`（和 ChatGarment 互動的網站）、`*.html`（網格檢視、比較頁）、`experiments/`（各次實驗，說明寫在每支程式開頭）。

## 跑法

在 repo 根目錄：

```bash
.venv\Scripts\python flows\body_fit\export_neutral.py results\chatgarment_ai\bf16_3090
```

這台筆電的路徑有中文（「桌面」），libigl 會開不了檔：要縫合、讀寫網格時，從 `D:\newfasion_VT_work\repo_ascii` 的英文路徑副本執行。
