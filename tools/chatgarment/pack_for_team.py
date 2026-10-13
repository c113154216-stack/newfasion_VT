# -*- coding: utf-8 -*-
"""把 export_for_physics.py 的一件衣服打包給隊友試用（只有網格，不含顏色花紋）。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\pack_for_team.py [衣服名稱，預設 e01_red_tee_r2]

輸出：output/team_share/<名稱>/（資料夾）和 output/team_share/<名稱>.zip
內容：衣服 OBJ（拿掉材質那行）、固定點 JSON、對照用 T-pose 身體 OBJ、預覽圖、README.md
"""
import json
import shutil
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from gallery import draw, shaded  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output" / "physics_export"
DST = ROOT / "output" / "team_share"
BODY = "tpose_smpl_female.obj"


def preview(gv, gf, bv, bf, info, path):
    gv, bv = gv * 100, bv * 100  # 畫圖用公分
    bf = bf[bv[bf].mean(axis=1)[:, 1] > 70]
    fig = plt.figure(figsize=(9, 4.6))
    for j, (azim, view) in enumerate([(90, "正面"), (0, "側面")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        draw(ax, [(bv, bf, shaded(bv, bf, np.array([0.80, 0.66, 0.54]))),
                  (gv, gf, shaded(gv, gf, np.tile([[0.62, 0.64, 0.68]], (len(gf), 1))))])
        p = gv[info["pins"]["neck_shoulder_band"]]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], c="red", s=8, depthshade=False)
        ax.set_xlim(-85, 85); ax.set_ylim(-85, 85); ax.set_zlim(60, 175)
        ax.view_init(5, azim); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
        ax.set_title(view, fontsize=11)
    fig.suptitle(f"紅＝固定點 neck_shoulder_band（{len(info['pins']['neck_shoulder_band'])} 點）", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def readme(name, info, gv):
    lo, hi = gv.min(axis=0), gv.max(axis=0)
    size = (hi - lo) * 100
    return f"""# 衣服網格試用：{name}

由商品照片 → ChatGarment 判斷款式 → GarmentCode 打版 → 幾何縫合 產生。只有形狀，還沒有顏色花紋。

## 檔案
| 檔案 | 說明 |
|---|---|
| `{name}.obj` | 衣服網格（縫合好的一整件，三角形） |
| `{name}_pins.json` | 固定點：領口、肩線的頂點編號 |
| `{BODY}` | 對照用身體（T-pose），衣服就是照這個身體縫的 |
| `preview.png` | 預覽圖 |

## 座標與格式
- 單位：**公尺**；+y 朝上、+z 朝前（人面向 +z）、腳底在 y = 0
- 姿勢：**T-pose**（手臂水平）
- 頂點 {info['n_vertices']:,}、三角形 {info['n_faces']:,}，頂點間距約 {info['resolution_cm']:g} cm
- 衣服範圍：寬 {size[0]:.0f} × 高 {size[1]:.0f} × 厚 {size[2]:.0f} cm（含袖子）
- OBJ 有 UV（`vt`），沒有法線、沒有蒙皮權重
- 固定點編號**從 0 開始**，對應 OBJ 裡第幾個 `v`（第一個 `v` 是 0）
  - `neck_shoulder_band`（**建議用這組**，{len(info['pins']['neck_shoulder_band'])} 點）：領口到肩線一圈。
    規則和物理端自動推的一樣：最近身體面由頸與左右鎖骨主導（SMPL 蒙皮分區），且離身體 1 cm 內
  - `neckline`：領口一圈；`shoulder`：肩線縫線；`neckline_and_shoulder`：兩者合併（只有線，較少點）

## 打開方式
- 直接拖進 Blender（File → Import → Wavefront .obj，Forward 選 Z、Up 選 Y）或 https://3dviewer.net
- Three.js：`OBJLoader` 讀進來單位是公尺，不用縮放

## 注意
- 身體是 GarmentCode 附的 SMPL 女性平均身材，**不是**我們專題 SMPL-X 生成的身體；
  換成 SMPL-X 身體前，衣服可能和身體有幾公分的偏差。
- 衣長是 AI 從照片估的，可能偏短或偏長，之後會用尺寸表校正。
"""


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "e01_red_tee_r2"
    src = SRC / name
    out = DST / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # OBJ 拿掉 mtllib：隊友那邊沒有貼圖檔，留著會出現找不到材質的警告
    lines = (src / f"{name}.obj").read_text(encoding="utf-8").splitlines(keepends=True)
    (out / f"{name}.obj").write_text("".join(l for l in lines if not l.startswith(("mtllib", "usemtl"))),
                                     encoding="utf-8")
    shutil.copy(SRC / "_bodies" / BODY, out / BODY)

    info = json.loads((src / f"{name}_pins.json").read_text(encoding="utf-8"))
    info["body_obj"] = BODY  # 改成包內的相對路徑
    info["note"] = "固定點編號從 0 起算，對應 OBJ 第幾個 v"
    (out / f"{name}_pins.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    gv, gf = igl.read_triangle_mesh(str(out / f"{name}.obj"))
    bv, bf = igl.read_triangle_mesh(str(out / BODY))
    preview(gv, gf, bv, bf, info, out / "preview.png")
    (out / "README.md").write_text(readme(name, info, gv), encoding="utf-8")

    zip_path = shutil.make_archive(str(DST / name), "zip", root_dir=DST, base_dir=name)
    print(out)
    print(zip_path)


if __name__ == "__main__":
    main()
