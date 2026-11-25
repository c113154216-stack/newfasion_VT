# size_chart：照廠商尺寸表打版

照商品頁的尺寸表（平放量的 cm）直接打版，不靠 AI 的長度、寬度參數；每個尺碼（S／M／L…）各做一件，縫到同一個身體上比較。
款式（領型、袖型、袖口…）仍沿用 `body_fit` 流程的 design，只改尺寸相關的參數。

| 尺寸表欄位 | 改哪個參數 | 怎麼改 |
|---|---|---|
| 胸寬 | `shirt.width` | 胸寬 × 2 ÷ 身體胸圍，公式直接算 |
| 衣長 | `shirt.length` | 在 2D 版片上量後片「肩頸點到下襬」，按比例修到差 0.2 cm 內 |
| 袖長 | `sleeve.length` | 在 2D 版片上量袖片「肩點到袖口」，按比例修到差 0.2 cm 內 |
| 肩寬 | 打版用身體的 `shoulder_w` | 比身體寬就做成落肩：上衣換成落肩版型 `DropShoulderShirt`（`drop_shoulder.py`，袖籠是腋下到肩點的斜直線，肩點可以超出側縫）、肩斜放平、袖子角度 0、袖山不抽皺。見 `spec_batch.py` 的 `fit2` |

整批（`spec_batch.py`）照上面打好版就縫一次，3D 只量、不回頭修版片（2026-11-24 起）。
3D 量法在 `spec_measure.py`：肩點用 GarmentCode 的縫線資料找（後片袖籠縫線最上端），距離用精確的布面最短距離。
驗證過程和圖在 `docs/驗證_尺寸/`。

## 程式（照順序跑）

| 程式 | 做什麼 | 輸出 |
|---|---|---|
| `spec_to_pattern.py` | 照尺寸表打版，只在 2D 版片上量，每輪不到 1 秒 | `output/spec_to_pattern/<衣服>/` 各尺碼的 design、版片圖 |
| `spec_sew_compare.py` | 縫到 SMPL-X 中性身體上，比較三種肩寬 | `縫合_<尺碼>.png` |
| `spec_sizes.py` | 多個尺碼一起縫（預設 S、M、L）並排比較 | `尺碼_<S-M-L>.png`、`尺碼/<尺碼>/` |
| `spec_flatlay.py` | 各尺碼攤平成 2D 疊在一起，像商品平拍照 | `平放_<S-M-L>.png` |
| `spec_batch.py` | 整批：7 件網路衣服、每個尺碼打版＋縫合＋量 3D | `_batch/<代號>/<尺碼>/`（`.obj`、`量測.json`） |
| `spec_measure.py` | 3D 量測點（肩點、肩頸點）和精確布面距離，`spec_batch.py` 用 | — |
| `drop_shoulder.py` | 落肩版型 `DropShoulderShirt`（GarmentCode 原本的檔案沒改，import 時註冊進 MetaGarment） | — |
| `experiments/drop_compare.py` | 舊做法（Shirt）和落肩版型的對照表 | `對照表.json` |
| `spec_batch_figure.py`、`spec_batch_export.py` | 整批出圖、匯出給 engine 和試穿頁 | |

在 repo 根目錄：

整批（2026-11-25 的 `_batch` 就是這樣跑的：落肩款用落肩版型、袖籠深 ×1.2）：

```bash
set DROP_ARMHOLE_SCALE=1.2
.venv\Scripts\python flows\size_chart\spec_batch.py --drop b
.venv\Scripts\python flows\size_chart\spec_batch.py --summary
.venv\Scripts\python flows\size_chart\spec_batch_figure.py
.venv\Scripts\python flows\size_chart\spec_batch_export.py C:/Users/User/Desktop/VTO_PROJ
```

`--drop b`：判定是落肩款就用落肩版型（`--drop a`＝Shirt 推不到才換；不加＝全部用 Shirt）。`DROP_ARMHOLE_SCALE` 程式預設還是 1.0，1.2 是從 z04 M、z06 M 試出來的（還沒定成預設）。

單件、單尺碼：

```bash
.venv\Scripts\python flows\size_chart\spec_to_pattern.py flows\size_chart\網路衣服_尺寸資料\z06_長袖T
.venv\Scripts\python flows\size_chart\spec_sizes.py flows\size_chart\網路衣服_尺寸資料\z06_長袖T S,M,L
```

款式目前借 `body_fit` 匯出的 i10 紅色圓領長袖 T（`output/physics_export/neutral/i10_red_crew_long_sleeve_tee/design_used.yaml`），要先跑過 `body_fit/export_neutral.py`。
打版、縫合直接用 `body_fit/` 的程式（`design_to_mesh.py`、`export_neutral.py` 的 `build()`）。

## 資料

[`網路衣服_尺寸資料/`](網路衣服_尺寸資料/README.md)：ZOZOTOWN 7 件衣服的尺寸表（`尺寸.json`、`全部尺寸.json`）。商品照只放本機，不上傳。
