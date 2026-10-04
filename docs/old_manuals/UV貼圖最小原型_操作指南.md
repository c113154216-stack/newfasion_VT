# UV 貼圖最小原型 — 操作指南

這份文件記錄「把商品照片印到 3D 服裝模型上」這件事的最小驗證流程，包含完整可執行的程式碼、
怎麼準備照片、怎麼跑、怎麼在網站上看結果，以及過程中踩到的坑跟怎麼修。目標是讓你照著這份文件
自己重現整個流程，不用重新推導一次。

對應的實際檔案：`3Dclothes_project/_uv_prototype_test.py`（獨立驗證腳本，不影響任何生產程式碼）。

---

## 一、整體流程（五個步驟）

```
1. 準備材料：讀商品照片 + 讀取服裝模型（現有的 .obj 檔案或程序化生成的網格）
2. 判斷每個頂點是不是「正面」（用 Z 座標，Z >= 0 視為正面）
3. 正面的點：算出 UV 座標（正交投影，(x,y) 線性縮放到 0~1）
4. 非正面的點：UV 固定指向照片正中央（衣服本身的顏色，不是背景）
5. 把 UV + 照片綁定到模型上，輸出成 .glb 檔案
```

---

## 二、照片怎麼準備

**最小驗證階段不需要真的分割照片**，只要準備「一張」照片即可，但要注意幾點：

1. **格式**：`.jpg` / `.png` 都可以，程式碼用 `PIL.Image.open(path).convert("RGB")` 讀取，會自動統一成 RGB 三色格式。
2. **內容要求**（這是這次踩坑學到的重點）：
   - 照片要**乾淨、置中、單一主體**——不要用拼接多張人像的複合照片（會導致貼出來一半是雜訊）
   - 商品照片背景通常是白色/淺色，**不會影響最小驗證的正確性判斷**，但如果要看清楚有沒有貼對，
     建議先用**高對比的測試圖**（例如棋盤格）驗證技術路徑，最後才換成真正的商品照
3. **驗證用棋盤格圖怎麼生成**（業界標準做法，比直接用商品照片更容易判斷對不對）：

```python
from PIL import Image, ImageDraw

size = 512
grid = 8
cell = size // grid
img = Image.new('RGB', (size, size), (255, 255, 255))
draw = ImageDraw.Draw(img)

colors = [(230,57,70), (69,123,157), (244,196,48), (42,157,143), (155,93,229), (247,127,0)]
for row in range(grid):
    for col in range(grid):
        x0, y0 = col*cell, row*cell
        x1, y1 = x0+cell, y0+cell
        c = colors[(row+col) % len(colors)] if (row+col) % 2 == 0 else (20,20,20)
        draw.rectangle([x0,y0,x1,y1], fill=c)

# 加外框 + 十字線 + 方向標記，方便判斷貼出來有沒有翻轉、鏡像
draw.rectangle([0,0,size-1,size-1], outline=(0,255,0), width=6)
draw.line([size//2,0,size//2,size], fill=(0,255,0), width=4)
draw.line([0,size//2,size,size//2], fill=(0,255,0), width=4)

img.save('checker_test.png')
```

**為什麼棋盤格比商品照片更適合測試：** 每一格顏色差異巨大，UV 有沒有拉伸、扭曲、方向錯誤，
一眼就看得出來；不像商品照片，貼歪了也不一定明顯看得出來。

---

## 三、完整程式碼（逐段解釋）

### 3.1 讀取材料

```python
import sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "geometry_physics"))
import geometry_engine as ge

image = Image.open(PHOTO_PATH).convert("RGB")

# 兩種模型來源擇一：
# (A) 程序化生成的軀幹（快速測試用，不含袖子）
vertices, weights, faces = ge._build_torso()
# (B) 真正商品在用的模型檔案（更貼近實際上架效果）
vertices, weights, faces = ge.build_real_shirt_template(str(GARMENT_PATH))
```

**這一步只管「把材料擺出來」**，照片跟模型形狀是兩件互不相干的事，先各自準備好。

### 3.2 判斷正面 + 算 UV（核心函式）

```python
def compute_uv(vertices: np.ndarray) -> np.ndarray:
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # 關卡一：只看模型自己的 Z 座標，跟照片完全無關
    is_front = z >= 0.0

    uv = np.zeros((len(vertices), 2), dtype=np.float64)

    # 關卡二：正面的點，各自算出獨立的 UV（正交投影）
    front_x, front_y = x[is_front], y[is_front]
    x_min, x_max = front_x.min(), front_x.max()
    y_min, y_max = front_y.min(), front_y.max()
    u_front = (front_x - x_min) / (x_max - x_min)
    v_front = (front_y - y_min) / (y_max - y_min)
    uv[is_front, 0] = u_front
    uv[is_front, 1] = v_front

    # 關卡三：非正面的點，全部固定指向照片正中央（抓衣服本身的顏色，不是背景角落）
    uv[~is_front] = [0.5, 0.5]

    return uv
```

**三個關卡的判斷順序，一定要先「分類」才能「算 UV」**——因為非正面的點根本不需要算連續的 UV，
直接給固定值即可。

### 3.3 綁定材質 + 輸出檔案

```python
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# 踩坑修正一：面朝向可能是反的（法向量朝內），會導致貼圖出現在「內壁」而不是外壁
mesh.fix_normals()

mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)

# 踩坑修正二：trimesh 預設材質是全金屬（metalness=1），
# 沒有環境貼圖時幾乎看不出貼圖顏色，手動改成非金屬材質
mesh.visual.material.metallicFactor = 0.0
mesh.visual.material.roughnessFactor = 0.85

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
mesh.export(str(OUTPUT_PATH))
```

---

## 四、怎麼執行

```bash
cd 3Dclothes_project
python _uv_prototype_test.py
```

跑完會在 `data/outputs/_uv_prototype_test.glb` 產生輸出檔案，並印出頂點數、正面頂點比例、
檔案大小等資訊，方便確認有沒有跑成功。

---

## 五、怎麼在網站上看結果（不用重新整合進正式流程）

1. 啟動 MySQL（XAMPP `mysqld.exe`）+ Flask（`flask --app app run`）
2. 瀏覽器開 `http://127.0.0.1:5000/dev`，捲到「5. 試穿」那一區塊
3. 打開瀏覽器開發者工具（F12）→ Console，貼上：

```javascript
const mod = await import("/web_test/tryon.js");
await mod.loadTryOnGlb("/static/outputs/_uv_prototype_test.glb?t=" + Date.now());
```

**重點：每次重新產生 glb 檔案後，要重新貼一次這段指令（換一個新的時間戳記），不然瀏覽器可能會
用到舊的快取版本。** 這裡用的 `loadTryOnGlb()` 是網站正式程式碼裡的函式，不是另外寫的測試邏輯，
所以這樣測出來的結果，跟真正整合進生產流程的效果是一致的。

---

## 六、過程中踩到的坑（照時間順序）

| # | 問題 | 症狀 | 原因 | 修法 |
|---|---|---|---|---|
| 1 | 畫面全黑/太暗看不出貼圖 | 顏色統一偏暗灰 | trimesh 匯出材質預設 `metalness=1`，沒環境貼圖時幾乎不顯示貼圖顏色 | `mesh.visual.material.metallicFactor = 0.0` |
| 2 | 貼圖看起來很單調、沒有明顯花紋 | 顏色跟著照片主色調變，但看不出細節 | 測試用的照片本身內容不適合（複合拼接照、低對比白底照） | 換成高對比棋盤格測試圖 |
| 3 | 顏色出現在「內壁」 | 從外面看穿過模型，看到內側被貼上顏色 | 面的朝向（winding）反了，法向量指向內部 | `mesh.fix_normals()` |

**關於第 3 個坑的補充**：`mesh.fix_normals()` 這個自動修正，在**程序化生成的軀幹網格**
（`_build_torso()`，頭尾開放、不封閉的管狀網格）上測試時效果不明顯——推測是因為這個自動修正演算法
仰賴「封閉網格」的體積判斷邏輯來決定朝向，對開放管狀網格可能失效。但在**真正商品在用的模型檔案**
（`build_real_shirt_template()` 讀取的 `tshirt_base.obj`，結構更完整）上測試，同一行程式碼
**確實修正成功**，貼圖正確顯示在外壁。這代表如果之後要換其他程序化生成的簡化網格測試，
面朝向問題可能還是要另外處理（例如手動確認 `_grid_faces()` 的頂點連接順序），但在真正的商品模型上
不用擔心這個問題。

---

## 七、目前狀態

- UV 座標計算邏輯：已多次獨立驗證正確（Python 端直接繪圖確認、trimesh 讀回比對、embedded 材質比對）
- 材質綁定 + glb 匯出：已驗證正確
- 面朝向問題：**在真正商品模型（`tshirt_base.obj`）上已確認 `fix_normals()` 修正有效**，貼圖正確顯示在外壁
- **端到端完整驗證成功**：用真正商品模型 + 棋盤格測試圖，跑完整條流程（讀模型 → 判斷正面 → 算 UV →
  修正朝向 → 綁定材質 → 修正金屬感 → 輸出 glb），在獨立檢視頁面上確認結果正確——形狀完整（含袖子）、
  棋盤格清楚顯示在外壁、方塊比例沒有明顯扭曲
- 尚未做：正式整合進 `geometry_engine.py` / `mesh_glb.py` 生產程式碼（目前都在獨立測試腳本
  `_uv_prototype_test.py` 裡驗證，沒有動到任何正式檔案）；也還沒把測試圖換回真正的商品照片重新驗證一次
- **待確認（⚠️ 優先度高）**：`geometry_physics/assets/garments/tshirt_base.obj` 等五個 `.obj` 檔案
  查無來源紀錄——檔案本身沒有作者/授權註解，git 只有一次性的 initial commit，團隊文件裡也只寫
  「怎麼用」沒寫「哪裡來」。`build_real_shirt_template()` 的註解提到是為了讀外部資料集（例如 MGN
  Multi-Garment）設計的，但無法確認現有這五個檔案是不是真的來自該資料集，還是紀泓宇自己建模。
  **要跟紀泓宇確認來源跟授權條款，尤其如果之後有對外展示或提交的情況。**
