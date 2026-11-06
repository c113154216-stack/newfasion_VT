# -*- coding: utf-8 -*-
"""
袖口縮口試驗（cuff_test.py）的對比圖與量測：每個版本量袖口一圈、該處手臂一圈、袖口離手腕，畫全身＋手臂特寫。

跑法（在 newfashion_VT 根目錄，先跑過 cuff_test.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\cuff_figure.py
輸出：output/chatgarment/袖口縮口_g03.png
"""
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "chatgarment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gallery import draw, panel_colors, shaded  # noqa: E402
from sleeve_sew_ablation import arm  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "output" / "chatgarment"
VERSIONS = [("aibf16_3090_v2_g03_blue_hoodie", "原本（沒有縮口）"),
            ("aibf16_cuff_g03_blue_hoodie", "加縮口，top_ruffle 1"),
            ("aibf16_cuff_g03_blue_hoodie_r1.7", "加縮口，top_ruffle 1.7")]


def measure(name, bv, bf, body):
    """左袖：袖口一圈、該處手臂一圈（截面凸包）、袖口離手腕。"""
    d = OUT / name
    v, f = igl.read_triangle_mesh(str(d / f"{name}_sewn.obj"))
    seg = (d / f"{name}_sim_segmentation.txt").read_text().splitlines()
    c, u, wrist = arm(bv, -1)
    for loop in igl.all_boundary_loop(f):
        loop = np.array(loop)
        if v[loop, 0].mean() > 0 or np.mean([("cuff" in seg[i] or "sleeve" in seg[i]) for i in loop]) < 0.6:
            continue
        ring = np.linalg.norm(np.roll(v[loop], -1, 0) - v[loop], axis=1).sum()
        cen = v[loop].mean(0)
        sec = min(body.section(plane_origin=cen, plane_normal=u).discrete, key=lambda p: np.linalg.norm(p.mean(0) - cen))
        flat = (sec - sec.mean(0)) @ np.linalg.svd(sec - sec.mean(0))[2][:2].T
        end = ((v[loop] - c) @ u).max() - wrist
        return v, f, ring, ConvexHull(flat).area, end, c + u * wrist
    raise ValueError(f"{name}：找不到左袖口")


def main():
    bv, bf = igl.read_triangle_mesh(str(ROOT / "third_party" / "GarmentCodeRC" / "assets" / "bodies" / "mean_male.obj"))
    bv *= 100
    body = trimesh.Trimesh(bv, bf, process=False)
    bf2 = bf[bv[bf].mean(axis=1)[:, 1] > 60]
    bc = shaded(bv, bf2, np.array([0.80, 0.66, 0.54]))
    fig = plt.figure(figsize=(4 * (1 + 2 * len(VERSIONS)) * 0.8, 6.5))
    ax = fig.add_subplot(1, 1 + 2 * len(VERSIONS), 1)
    ax.imshow(Image.open(ROOT / "output" / "chatgarment_ai" / "inputs_v2" / "g03_blue_hoodie.png"))
    ax.axis("off"); ax.set_title("g03 照片", fontsize=13)
    for k, (name, label) in enumerate(VERSIONS):
        v, f, ring, armc, end, wr = measure(name, bv, bf, body)
        print(f"{label}：袖口一圈 {ring:.1f} cm、該處手臂 {armc:.1f} cm、袖口離手腕 {end:+.1f} cm")
        col = shaded(v, f, panel_colors(OUT / name, name, f))
        for j in range(2):
            ax = fig.add_subplot(1, 1 + 2 * len(VERSIONS), 2 + 2 * k + j, projection="3d")
            draw(ax, [(bv, bf2, bc), (v, f, col)])
            ax.computed_zorder = False
            ax.scatter(wr[0], wr[2], wr[1], s=70, c="red", depthshade=False)
            if j == 0:
                ax.set_xlim(-55, 55); ax.set_ylim(-55, 55); ax.set_zlim(65, 175)
                ax.set_title(f"{label}\n袖口 {ring:.0f} cm／手臂 {armc:.0f} cm\n離手腕 {end:+.1f} cm", fontsize=11)
            else:
                ax.set_xlim(wr[0] - 10, wr[0] + 30); ax.set_ylim(wr[2] - 20, wr[2] + 20); ax.set_zlim(wr[1] - 10, wr[1] + 30)
                ax.set_title("左手特寫", fontsize=11)
            ax.view_init(8, 90); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
    fig.suptitle("藍帽 T：羅紋縮口（紅點＝手腕）", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    dst = OUT / "袖口縮口_g03.png"
    plt.savefig(dst, dpi=70)
    print(dst)


if __name__ == "__main__":
    main()
