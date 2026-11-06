# -*- coding: utf-8 -*-
"""
有／無壓短限制的對照圖（實驗三 sleeve_compress_test.py 的結果），三張：
  壓短限制_1_整件.png    每件衣服左右兩張：新縫合（現在）vs ＋壓短限制 2%，紅點＝手腕
  壓短限制_2_被壓短的地方.png  衣服依「邊被壓到原長幾成」上色：越紅＝被壓越短
  壓短限制_3_袖子特寫.png    右袖放大、上下疊著比，看長度差多少

跑法（在 newfashion_VT 根目錄，先跑過 sleeve_compress_test.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\compress_limit_figures.py
輸出：output/chatgarment/壓短限制_*.png
"""
import json
import pickle
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from body_compare_figure import BODIES, NEUTRAL, SKIN, load, shade  # noqa: E402
from sew import mesh_edges, rest_lengths  # noqa: E402
from sleeve_compress_test import GARMENTS, OUT  # noqa: E402
from sleeve_smplx_test import arms_from_joints  # noqa: E402

ROOT = HERE.parents[2]
DST = ROOT / "output" / "chatgarment"
BOX = ROOT / "output" / "chatgarment"  # design_to_mesh 的輸出（smplx_nocuff_<衣服>/）
CASES = [("new", "沒有壓短限制（現在的新縫合）"), ("c02", "＋壓短限制 2%")]
ORANGE = "#e8833a"
CMAP = plt.get_cmap("RdYlBu")  # 紅＝被壓短，藍＝原長
NORM = Normalize(0.8, 1.0)


def face_ratio(x, f, E, rest):
    """每個面上「被壓最短的那條邊」是原長的幾成。"""
    ratio = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1) / rest
    vmin = np.ones(len(x))
    np.minimum.at(vmin, E[:, 0], ratio)
    np.minimum.at(vmin, E[:, 1], ratio)
    return vmin[f].min(axis=1)


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    body, bf = load(BODIES / "smplx_neutral_theta0_feet0.obj", 100)
    lift = -igl.read_triangle_mesh(str(BODIES / "smplx_neutral_theta0.obj"))[0][:, 1].min()
    wrists = [c + u * w for c, u, w in arms_from_joints(lift).values()]
    label = dict(CASES)

    data = {}
    for g in GARMENTS:
        short = g.split("_v2_")[1]
        name = f"smplx_nocuff_{short}"
        v0, f = igl.read_triangle_mesh(str(BOX / name / f"{name}_boxmesh.obj"))
        E = mesh_edges(f)
        rest = rest_lengths(v0, E, pickle.loads((BOX / name / f"{name}_orig_lens.pickle").read_bytes()))
        seg = (BOX / name / f"{name}_sim_segmentation.txt").read_text().splitlines()
        sleeve = np.array(["sleeve" in s for s in seg])
        data[short] = {code: load(OUT / f"{code}_{short}_sewn.obj", 1)[0] for code, _ in CASES}
        data[short]["f"], data[short]["E"], data[short]["rest"], data[short]["sleeve"] = f, E, rest, sleeve

    def dots(ax):
        for p in wrists:
            ax.scatter(p[0], p[1], s=70, c="red", zorder=5)

    # 圖一：整件
    fig, axes = plt.subplots(len(GARMENTS), 2, figsize=(16, 5.6 * len(GARMENTS)))
    for r, (short, d) in enumerate(data.items()):
        for c, (code, title) in enumerate(CASES):
            ax = axes[r, c]
            shade(ax, [(body, bf, SKIN), (d[code], d["f"], NEUTRAL if code == "new" else ORANGE)])
            dots(ax)
            m = results[short][[k for k in results[short] if k.endswith(("（現在）", "2%")) and
                                ("壓短" in k) == (code == "c02") and not k.startswith("縮口")][0]]
            ax.set_xlim(-92, 92); ax.set_ylim(78, 180); ax.grid(alpha=0.25)
            ax.set_title(f"{short}　{title}\n袖口離手腕 {m['袖口離手腕cm']:+.1f} cm", fontsize=14)
    fig.suptitle("有無壓短限制：整件比較（T-pose、SMPL-X 中性；紅點＝手腕）", fontsize=17)
    plt.tight_layout()
    plt.savefig(DST / "壓短限制_1_整件.png", dpi=70)
    plt.close(fig)

    # 圖二：被壓短的地方
    fig, axes = plt.subplots(len(GARMENTS), 2, figsize=(16, 5.6 * len(GARMENTS)))
    for r, (short, d) in enumerate(data.items()):
        for c, (code, title) in enumerate(CASES):
            ax = axes[r, c]
            fr = face_ratio(d[code], d["f"], d["E"], d["rest"])
            shade(ax, [(body, bf, SKIN), (d[code], d["f"], CMAP(NORM(fr))[:, :3])])
            dots(ax)
            s_e = d["sleeve"][d["E"][:, 0]] & d["sleeve"][d["E"][:, 1]]
            r_e = np.linalg.norm(d[code][d["E"][s_e, 0]] - d[code][d["E"][s_e, 1]], axis=1) / d["rest"][s_e]
            ax.set_xlim(-92, 92); ax.set_ylim(78, 180); ax.grid(alpha=0.25)
            ax.set_title(f"{short}　{title}\n袖子被壓到原長 9 成以下的邊：{100 * np.mean(r_e < 0.9):.0f}%", fontsize=14)
    fig.colorbar(ScalarMappable(NORM, CMAP), ax=axes, shrink=0.6, label="邊長 ÷ 原長（紅＝被壓短）")
    fig.suptitle("哪裡被壓短了？（顏色＝該處被壓最短的邊是原長幾成）", fontsize=17)
    plt.savefig(DST / "壓短限制_2_被壓短的地方.png", dpi=70, bbox_inches="tight")
    plt.close(fig)

    # 圖三：右袖特寫（x > 0 那隻），兩種縫法疊在一起
    fig, axes = plt.subplots(len(GARMENTS), 1, figsize=(11, 5.8 * len(GARMENTS)))
    for r, (short, d) in enumerate(data.items()):
        ax = axes[r]
        keep = lambda x: d["f"][(x[d["f"]][:, :, 0] > 12).all(axis=1)]  # 只畫右半邊
        shade(ax, [(body, bf[(body[bf][:, :, 0] > 12).all(axis=1)], SKIN)])
        for k, (code, title) in enumerate(CASES):
            x = d[code] + [0, 22 * (1 - k), 0]  # 沒限制的往上移 22 cm，上下對照
            shade(ax, [(x, keep(d[code]), NEUTRAL if code == "new" else ORANGE)])
            tip = x[d["sleeve"] & (x[:, 0] > 0), 0].max()
            ax.axvline(tip, color=NEUTRAL if code == "new" else ORANGE, ls="--", lw=1.5)
            low = x[d["sleeve"] & (x[:, 0] > 0), 1].min()
            ax.text(tip - 1, low - 3, f"{'沒限制' if code == 'new' else '有限制'}：袖口 x＝{tip:.0f}", fontsize=12, ha="right", va="top",
                    color=NEUTRAL if code == "new" else ORANGE)
        wx = max(p[0] for p in wrists)
        ax.axvline(wx, color="red", lw=2)
        ax.text(wx + 1, 112, "手腕", color="red", fontsize=12)
        ax.set_xlim(10, 92); ax.set_ylim(104, 172); ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_title(f"{short}　右袖放大\n上：沒限制（往上移 22 cm 方便比較）　下：加壓短限制 2%　虛線＝袖口", fontsize=13)
    plt.tight_layout()
    plt.savefig(DST / "壓短限制_3_袖子特寫.png", dpi=70)
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
