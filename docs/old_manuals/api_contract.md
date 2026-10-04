# 3D 虛擬試穿 — API 接口契約（v1）

> 維護者：趙丞章  
> 消費者：林宇鑫（前端 / FashionCLIP）、紀泓宇（M2～M4）

## 統一回應格式

```json
{ "success": true, "data": { ... } }
{ "success": false, "error": "message" }
```

## REST 端點

### GET /api/health

健康檢查。

**Response 200**

```json
{ "success": true, "data": { "status": "ok" } }
```

---

### POST /api/products/upload

商家上架（multipart/form-data）。

| 欄位 | 類型 | 必填 |
|------|------|------|
| name | string | 是 |
| category | string | 是 |
| description | string | 否 |
| size_chart | JSON string | 否 |
| garment_template_id | string | 否 |
| image | file | 是 |

**Response 201**

```json
{
  "success": true,
  "data": {
    "product_id": 1,
    "name": "白色短袖",
    "faiss_vector_id": 0,
    "image_url": "/static/uploads/products/1.jpg"
  }
}
```

---

### GET /api/products/{id}

商品詳情。

---

### POST /api/recommend

穿搭推薦（JSON）。

**Request**

```json
{
  "query_text": "白色短袖",
  "query_image": null,
  "top_k": 10
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "product_id": 1,
        "name": "白色短袖",
        "category": "top",
        "score": 0.92,
        "image_url": "/static/uploads/products/1.jpg"
      }
    ]
  }
}
```

---

### POST /api/tryon/simulate

觸發 3D 試穿（同步 v1）。

**Request**

```json
{
  "product_id": 1,
  "size": "M",
  "body": {
    "height_cm": 175,
    "chest_cm": 92,
    "waist_cm": 78,
    "hip_cm": 96
  }
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "session_id": 42,
    "result_glb_url": "/static/outputs/42.glb",
    "status": "done",
    "elapsed_seconds": 0.05
  }
}
```

---

### GET /api/tryon/status/{session_id}

查詢試穿狀態（非同步 / Celery 預留）。

**Response**

```json
{
  "success": true,
  "data": {
    "session_id": 42,
    "status": "done",
    "result_glb_url": "/static/outputs/42.glb",
    "error_message": null
  }
}
```

`status`: `pending` | `processing` | `done` | `failed`

---

## Adapter 接口（Python 內部）

| 模組 | 函式 | 負責人 | 輸入 | 輸出 |
|------|------|--------|------|------|
| FashionCLIP | `adapters.fashionclip_adapter.encode` | 林宇鑫 | `image_path`, `text` | `np.ndarray (512,)` |
| M2 | `adapters.body_generator.generate` | 紀泓宇 | `height_cm, chest_cm, waist_cm, hip_cm, output_name` | `{ "glb_path": str }` |
| M3 | `adapters.garment_processor.prepare` | 紀泓宇 | `product_id, size` | `{ "glb_path": str }` |
| M4 | `adapters.physics_simulator.run` | 紀泓宇 | `body_glb, garment_glb, output_name` | `{ "glb_path": str }` |

向量維度預設 **512**（Marqo-FashionCLIP），可透過環境變數 `FAISS_VECTOR_DIM` 調整。

## 靜態資源

| URL | 說明 |
|-----|------|
| `/static/uploads/products/{id}.jpg` | 商品圖 |
| `/static/outputs/{session_id}.glb` | 試穿結果 GLB |

## CORS

開發期允許：`http://localhost:5173`（Vite 預設），可於 `.env` 的 `CORS_ORIGINS` 調整。
