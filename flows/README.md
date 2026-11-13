# 衣物生成的兩條流程

同一個專案裡有兩種做衣服的方法，各放一個資料夾。打版、縫合、匯出的共用程式放在 `body_fit/`，`size_chart/` 直接引用它。

| 資料夾 | 流程 | 衣服大小由誰決定 | 用途 |
|---|---|---|---|
| [`body_fit/`](body_fit/README.md) | **照身體生成貼身衣物**（計畫書 2.4 節的主流程） | 使用者的身體＋版型餘量（AI 只決定款式） | 交付給 engine 的 68 件、試穿 |
| [`size_chart/`](size_chart/README.md) | **照廠商尺寸表打版** | 商品頁的尺寸表（胸寬、衣長、袖長；S／M／L 各一件） | 讓網格對得上真實成衣的號碼、比較不同尺碼 |

```
照片 ──► ChatGarment（款式）──► 照身體校正（領深、衣長、袖長）──► GarmentCode 打版 ──► 縫合 ──► 匯出      body_fit/
                                └─► 照尺寸表改胸寬、衣長、袖長 ──┘                                         size_chart/
```

兩條流程共用：
- 身體：SMPL-X 中性、θ＝0、β 全 0，打版尺寸檔 `mean_all.yaml`
- 打版：`body_fit/design_to_mesh.py`（GarmentCodeRC）
- 縫合：`body_fit/sew.py`、`body_fit/export_neutral.py` 的 `build()`
- 環境：GarmentCode 的 `.venv`（AI 推論另外用 `.venv-chatgarment`）

資料與輸出不在這裡：AI 推論結果在 `results/chatgarment_ai/`，產生的網格在 `output/`（不上傳）。
