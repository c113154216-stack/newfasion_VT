# -*- coding: utf-8 -*-
"""把所有縫好的衣服畫成一張總覽圖（正面，穿在對應的身體上）。

跑法（在 newfashion_VT 根目錄，先跑過 design_to_mesh.py、style_examples.py、sew.py）：
    .venv\\Scripts\\python tools\\chatgarment\\gallery.py

輸出：output/chatgarment/gallery.png
"""
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "chatgarment"
BODIES = REPO / "third_party" / "GarmentCodeRC" / "assets" / "bodies"

ITEMS = [
    ("shirt_mean_female", "mean_female", "短袖 T 恤・女性平均"),
    ("shirt_mean_male", "mean_male", "短袖 T 恤・男性平均"),
    ("shirt_mean_male_bust+12", "mean_male", "短袖 T 恤・男性胸圍 +12"),
    ("style_long_sleeve_fitted", "mean_male", "合身長袖上衣"),
    ("style_vneck_loose", "mean_male", "寬鬆 V 領短袖"),
    ("style_tank_top", "mean_male", "方領無袖背心"),
    ("style_oversize_long", "mean_male", "長版寬鬆 T 恤"),
    ("style_open_cardigan", "mean_male", "開襟長袖外套"),
    ("style_flare_blouse", "mean_male", "傘狀下襬短袖"),
]

LIGHT = np.array([0.35, 0.45, 1.0]) / np.linalg.norm([0.35, 0.45, 1.0])
PANEL_COLORS = ["#6b7fd7", "#8a5fa8", "#5aa0c8", "#c4687a", "#7fb38a", "#d49a5a", "#9a8fd0", "#5fb3b3"]


def shaded(v, f, base_rgb):
    """依面法向量和光線方向做簡單的明暗（雙面打光）。"""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    lit = 0.35 + 0.65 * np.abs(n @ LIGHT)
    return np.clip(base_rgb * lit[:, None], 0, 1)


def panel_colors(folder, name, f):
    """每片版片一個顏色，和 GarmentCode 的測試貼圖一樣用來分辨版片。"""
    labels = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
    panels = sorted({s for s in labels if not s.startswith("stitch")})
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in PANEL_COLORS])
    vid = np.array([panels.index(s) if s in panels else -1 for s in labels])
    face_panel = np.max(vid[f], axis=1) % len(rgb)
    return rgb[face_panel]


def draw(ax, parts):
    """parts 是 [(頂點, 面, 每面顏色), ...]。合成同一組一起畫，matplotlib 才會逐面排前後；
    分開畫的話它只依整組的平均深度排序，身體會整片蓋到衣服上。"""
    tris = np.concatenate([v[f][:, :, [0, 2, 1]] for v, f, _ in parts])
    colors = np.concatenate([c for _, _, c in parts])
    ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolor="none"))


def render(ax, name, body_name):
    folder = OUT / name
    sewn = folder / f"{name}_sewn.obj"
    ax.axis("off")
    if not sewn.exists():
        return
    bv, bf = igl.read_triangle_mesh(str(BODIES / f"{body_name}.obj"))
    bv = bv * 100
    gv, gf = igl.read_triangle_mesh(str(sewn))
    bf = bf[bv[bf].mean(axis=1)[:, 1] > 60]  # 身體只畫膝蓋以上，省時間
    draw(ax, [
        (bv, bf, shaded(bv, bf, np.array([0.80, 0.66, 0.54]))),
        (gv, gf, shaded(gv, gf, panel_colors(folder, name, gf))),
    ])
    ax.set_xlim(-55, 55); ax.set_ylim(-55, 55); ax.set_zlim(65, 175)
    ax.view_init(8, 90); ax.set_box_aspect((1, 1, 1))


def ai_gallery(prefix):
    """AI 推論的結果：每件衣服左邊是輸入照片，右邊是生成並縫好的網格。"""
    inputs = REPO / "output" / "chatgarment_ai" / "inputs"
    names = sorted(p.name[len(prefix) + 1:] for p in OUT.glob(f"{prefix}_*") if p.is_dir())
    rows = (len(names) + 1) // 2
    fig = plt.figure(figsize=(16, 4.2 * rows))
    for k, name in enumerate(names):
        photo = next(inputs.glob(f"{name}.*"), None)
        ax = fig.add_subplot(rows, 4, 2 * k + 1)
        if photo:
            ax.imshow(Image.open(photo).convert("RGB"))  # 有些 .png 其實是 JPEG，用 PIL 自動辨識
        ax.set_title(name, fontsize=12)
        ax.axis("off")
        render(fig.add_subplot(rows, 4, 2 * k + 2, projection="3d"), f"{prefix}_{name}", "mean_male")
    fig.suptitle(f"ChatGarment（{prefix}）：照片 → 款式參數 → GarmentCode 網格 → 縫合（男性平均身材、沒有重力）", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    path = OUT / f"gallery_{prefix}.png"
    plt.savefig(path, dpi=75)
    print(path)


def main():
    if len(sys.argv) > 1:
        return ai_gallery(sys.argv[1])
    fig = plt.figure(figsize=(15, 17))
    for k, (name, body_name, title) in enumerate(ITEMS):
        ax = fig.add_subplot(3, 3, k + 1, projection="3d")
        ax.set_title(title, fontsize=14)
        render(ax, name, body_name)

    fig.suptitle("縫合後的衣服網格（款式參數為人工指定，未經 AI；沒有重力，所以布不會下垂）", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "gallery.png"
    plt.savefig(path, dpi=85)
    print(path)


if __name__ == "__main__":
    main()
