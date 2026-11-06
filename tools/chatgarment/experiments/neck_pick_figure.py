# -*- coding: utf-8 -*-
"""
neck_pick_mesh.py 生成的網格和原始照片並排：左邊照片、右邊上半身正面特寫（看領口）。

跑法（在 newfashion_VT 根目錄，先跑過 neck_pick_mesh.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\neck_pick_figure.py
輸出：output/chatgarment/領口比對_挑8件.png（照片是官方商品照，只限內部看，不上傳）
"""
import json
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "chatgarment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gallery import draw, panel_colors, shaded  # noqa: E402
from neck_pick_mesh import PICKS, TAG  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "output" / "chatgarment"
PHOTOS = ROOT / "output" / "chatgarment_ai" / "inputs_i"
LABELS = json.loads((ROOT / "results" / "chatgarment_ai" / "bf16_i" / "labels.json").read_text(encoding="utf-8"))
SHAPE = {"CircleNeckHalf": "圓", "CurvyNeckHalf": "圓(曲)", "VNeckHalf": "V", "SquareNeckHalf": "方",
         "TrapezoidNeckHalf": "梯形", "CircleArcNeckHalf": "圓(弧)", "Bezier2NeckHalf": "圓(貝茲)"}
COMP = {None: "", "Turtle": "＋高領", "SimpleLapel": "＋翻領", "Hood2Panels": "＋帽子"}


def main():
    import yaml
    bv, bf = igl.read_triangle_mesh(str(ROOT / "third_party" / "GarmentCodeRC" / "assets" / "bodies" / "mean_male.obj"))
    bv *= 100
    bf = bf[bv[bf].mean(axis=1)[:, 1] > 110]  # 只畫上半身
    body_color = shaded(bv, bf, np.array([0.80, 0.66, 0.54]))
    n = len(PICKS)
    fig = plt.figure(figsize=(4 * 4, 4.4 * ((n + 1) // 2)))
    for k, g in enumerate(PICKS):
        name = f"{TAG}_{g}"
        folder = OUT / name
        design = yaml.safe_load((folder / "design_used.yaml").read_text(encoding="utf-8"))["design"]
        neck = json.loads((folder / "hem_fix.json").read_text(encoding="utf-8"))["領口校正"]
        made = SHAPE.get(design["collar"]["f_collar"]["v"], "?") + COMP[design["collar"]["component"]["style"]["v"]]
        depth = f"，前領深 {neck['實際深度 cm'][0]:.1f} cm" if neck and neck.get("實際深度 cm") else ""
        lab = LABELS[g]

        ax = fig.add_subplot((n + 1) // 2, 4, 2 * k + 1)
        ax.imshow(Image.open(PHOTOS / f"{g}.png")); ax.axis("off")
        ax.set_title(f"{g[:3]}　正確：{lab['組別領型']}", fontsize=12)

        ax = fig.add_subplot((n + 1) // 2, 4, 2 * k + 2, projection="3d")
        gv, gf = igl.read_triangle_mesh(str(folder / f"{name}_sewn.obj"))
        draw(ax, [(bv, bf, body_color), (gv, gf, shaded(gv, gf, panel_colors(folder, name, gf)))])
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_zlim(115, 175)
        ax.view_init(12, 90); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
        ax.set_title(f"做出：{made}{depth}", fontsize=12)
    fig.suptitle("第三批領口比對：照片 vs 生成的網格（款式修正＋領深校正＋新縫合）", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    dst = OUT / "領口比對_挑8件.png"
    plt.savefig(dst, dpi=70)
    print(dst)


if __name__ == "__main__":
    main()
