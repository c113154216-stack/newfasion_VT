# 系統架構文件

## ER 圖

```mermaid
erDiagram
    Product ||--o{ TryOnSession : has
    BodyProfile ||--o{ TryOnSession : uses

    Product {
        int id PK
        string name
        string category
        json size_chart
        string image_path
        string garment_template_id
        int faiss_vector_id
        string status
        datetime created_at
    }

    BodyProfile {
        int id PK
        float height_cm
        float chest_cm
        float waist_cm
        float hip_cm
        datetime created_at
    }

    TryOnSession {
        int id PK
        int product_id FK
        int body_profile_id FK
        string body_glb_path
        string result_glb_path
        string status
        string error_message
        datetime created_at
    }
```

## 架構圖

```mermaid
flowchart TB
    subgraph frontend [Frontend_Three.js]
        UI[UI]
    end

    subgraph backend [Flask_Backend]
        API[REST_API]
        M1[module1_pipeline]
        ORCH[orchestrator]
        VS[vector_store]
    end

    subgraph data [Data_Layer]
        MySQL[(MySQL_3d_tryon)]
        FAISS[(FAISS_index)]
        FS[Static_Files]
    end

    UI --> API
    API --> M1
    API --> ORCH
    M1 --> MySQL
    M1 --> VS
    VS --> FAISS
    ORCH --> FS
    M1 --> FS
```

## 效能策略（Phase 4）

1. 先量測 M2～M4 端到端耗時（`elapsed_seconds` 欄位）
2. **< 10 秒**：維持同步 `POST /api/tryon/simulate`
3. **> 30 秒**：引入 Celery + Redis，改用 `pending` + `GET /api/tryon/status/{id}` 輪詢

## 本地啟動

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m database.init_db
flask --app app run
```

XAMPP MySQL 需先建立 `3d_tryon` 資料庫（utf8mb4）。
