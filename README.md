# newfasion_VT

PHYS-VTO（結合參數化人體與布料物理模擬之低延遲虛擬試穿研究）中，林宇鑫負責的部分：

- 依本人體型與款式生成數位衣物（身體尺寸＋版型餘量，衣服不代表任何成衣號碼）
- 平面圖轉三維衣物模板
- Three.js 畫面：試穿、尺寸結論、推薦與搭配

## 試穿網站（`web/`）

純 HTML＋Three.js，不需要打包工具。目前商品、尺寸表、搭配都是假資料（`web/js/mock-data.js`），之後換成後端 API。

```bash
python -m http.server 8765 --directory web
```

開 http://localhost:8765 。鏡頭需要 localhost 或 HTTPS 才能開；沒開鏡頭時會跑示範動作。

| 檔案 | 內容 |
|---|---|
| `web/js/main.js` | 狀態、畫面組裝、事件 |
| `web/js/viewer.js` | Three.js 人偶與衣服（目前是簡化幾何，之後換 SMPL-X 與生成的衣服網格） |
| `web/js/pose.js` | Webcam＋MediaPipe Pose；示範動作 |
| `web/js/sizing.js` | 尺寸判定（計畫書表 2-1）與建議號碼 |
| `web/js/mock-data.js` | 假資料 |

## 目錄

| 路徑 | 內容 |
|---|---|
| `legacy/` | 從舊專案 `3Dclothes_project` 複製過來、可能用到的程式，保留原目錄結構，作為參考與起點 |
| `legacy/geometry_physics/` | SMPL-X 人體、衣服貼合、蒙皮權重（`body_model.py`）、部位縮放（`geometry_engine.py`） |
| `legacy/adapters/` | 舊版 M2～M4 adapter、GLB 匯出（含貼圖版） |
| `legacy/web/`、`legacy/web_test/` | 舊版產品站與 Three.js 試穿檢視頁 |
| `legacy/uv_learning/`、`legacy/_uv_prototype_*` | UV 貼圖原型 |
| `docs/old_manuals/` | 舊專案手冊（架構、API 契約、UV 工作記錄等） |

## 不在 repo 裡、需要自行放置的檔案

- **SMPL-X 權重**：`legacy/geometry_physics/assets/smplx/SMPLX_*.npz`，到 https://smpl-x.is.tue.mpg.de/ 註冊後下載（授權禁止公開散布）。
- **衣服模板 OBJ**：`legacy/geometry_physics/assets/garments/*.obj`，來源待確認（疑似 MGN 資料集），向組員取得。
