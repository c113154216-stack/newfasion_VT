# -*- coding: utf-8 -*-
"""把 design_to_mesh.py 和 sew.py 的結果畫成一張圖。
每一欄是一種身材；由上到下：2D 版片、縫合前（box mesh）、縫合後正面、縫合後背面。
縫合後的衣服依拉伸程度上色：藍＝接近原長，紅＝比版片拉長 10% 以上。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\preview.py
"""
import pickle
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sew import mesh_edges, rest_lengths  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "chatgarment"
BODIES = REPO / "third_party" / "GarmentCodeRC" / "assets" / "bodies"

# 胸圍 +12 的身體沒有對應的 3D 身體網格，用 mean_male 代替顯示（縫合時也是用它）
CASES = [
    ("mean_female", "mean_female", "女性平均"),
    ("mean_male", "mean_male", "男性平均"),
    ("mean_male_bust+12", "mean_male", "男性、胸圍 +12\n（身體網格仍用男性平均）"),
]


def face_stretch(x, faces, rest_of_edge):
    """每個三角形三條邊裡，最大的「目前長度／原始長度」。"""
    ratios = []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        e = np.sort(faces[:, [a, b]], axis=1)
        ratios.append(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1) / [rest_of_edge[tuple(k)] for k in e])
    return np.max(ratios, axis=0)


def draw(ax, v, f, colors, alpha, azim, max_faces=8000):
    idx = np.arange(len(f))[:: max(1, len(f) // max_faces)]
    c = colors if isinstance(colors, str) else colors[idx]
    ax.add_collection3d(Poly3DCollection(v[f[idx]][:, :, [0, 2, 1]], facecolor=c, edgecolor="none", alpha=alpha))
    ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_zlim(80, 170)
    ax.view_init(5, azim); ax.set_box_aspect((1, 1, 1)); ax.axis("off")


def main():
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(14, 19))
    for i, (name, body_name, title) in enumerate(CASES):
        folder = OUT / f"shirt_{name}"
        ax = fig.add_subplot(4, 3, i + 1)
        ax.imshow(plt.imread(next(folder.glob("*_pattern.png"))))
        ax.axis("off")
        ax.set_title(f"{title}\n2D 版片")

        bv, bf = igl.read_triangle_mesh(str(BODIES / f"{body_name}.obj"))
        bv = bv * 100  # 身體網格單位是公尺，衣服是公分
        box_v, f = igl.read_triangle_mesh(str(next(folder.glob("*_boxmesh.obj"))))
        sewn = next(folder.glob("*_sewn.obj"), None)

        ax = fig.add_subplot(4, 3, i + 4, projection="3d")
        draw(ax, bv, bf, "#d9b99b", 0.25, 90)
        draw(ax, box_v, f, "#9ca3af", 0.8, 90)
        ax.set_title("縫合前（box mesh）")
        if sewn is None:
            continue

        x, _ = igl.read_triangle_mesh(str(sewn))
        edges = mesh_edges(f)
        orig = pickle.loads(next(folder.glob("*_orig_lens.pickle")).read_bytes())
        rest = dict(zip(map(tuple, edges), rest_lengths(box_v, edges, orig)))
        stretch = face_stretch(x, f, rest)
        colors = cmap(np.clip((stretch - 1.0) / 0.2, 0, 1) * 0.5 + 0.5 * (stretch >= 1.0))
        for row, (azim, label) in enumerate([(90, "縫合後・正面"), (-90, "縫合後・背面")]):
            ax = fig.add_subplot(4, 3, i + 7 + 3 * row, projection="3d")
            draw(ax, bv, bf, "#d9b99b", 0.25, azim)
            draw(ax, x, f, colors, 0.95, azim)
            ax.set_title(f"{label}（紅＝拉長 >10%：{np.mean(stretch > 1.1) * 100:.0f}% 面）")

    plt.tight_layout()
    path = OUT / "preview.png"
    plt.savefig(path, dpi=80)
    print(path)


if __name__ == "__main__":
    main()
