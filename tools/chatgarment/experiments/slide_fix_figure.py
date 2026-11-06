# -*- coding: utf-8 -*-
"""
實驗四（slide_fix_test.py）的對比圖，兩張：
  上滑解法_1_整件.png  每列一件衣服、每欄一種縫法，紅點＝手腕，虛線＝縫合前的下襬
  上滑解法_2_數字.png  袖口離手腕、下襬上移、領口拉長、穿進身體的長條圖

跑法（在 newfashion_VT 根目錄，先跑過 slide_fix_test.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\slide_fix_figure.py
輸出：output/chatgarment/上滑解法_*.png
"""
import json
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

from body_compare_figure import BODIES, NEUTRAL, SKIN, load, shade  # noqa: E402
from slide_fix_test import GARMENTS, METHODS, OUT  # noqa: E402
from sleeve_smplx_test import arms_from_joints  # noqa: E402

DST = HERE.parents[2] / "output" / "chatgarment"
ORANGE = "#e8833a"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False
    res = {g: json.loads((OUT / f"results_{g}.json").read_text(encoding="utf-8")) for g in GARMENTS}
    body, bf = load(BODIES / "smplx_neutral_theta0_feet0.obj", 100)
    lift = -igl.read_triangle_mesh(str(BODIES / "smplx_neutral_theta0.obj"))[0][:, 1].min()
    wrists = [c + u * w for c, u, w in arms_from_joints(lift).values()]

    # 圖一：整件
    fig, axes = plt.subplots(len(GARMENTS), len(METHODS), figsize=(6 * len(METHODS), 5.4 * len(GARMENTS)))
    for r, g in enumerate(GARMENTS):
        for c, (code, (label, _)) in enumerate(METHODS.items()):
            ax = axes[r, c]
            gv, gf = load(OUT / f"{code}_{g}_sewn.obj", 1)
            shade(ax, [(body, bf, SKIN), (gv, gf, NEUTRAL if code == "now" else ORANGE)])
            for p in wrists:
                ax.scatter(p[0], p[1], s=50, c="red", zorder=5)
            ax.axhline(res[g]["縫合前"]["下襬離地cm"], color="k", ls="--", lw=1)
            ax.set_xlim(-92, 92); ax.set_ylim(72, 182); ax.grid(alpha=0.25)
            m = res[g][label]
            ax.set_title(f"{g.split('_')[0]}　{label}\n袖口離手腕 {m['袖口離手腕cm']:+.1f}　下襬上移 {m['下襬上移cm']:+.1f} cm\n"
                         f"領口拉長 {m['領口拉長%']:.0f}%　穿進身體 {m['穿進身體%']:.1f}%", fontsize=12)
    fig.suptitle("整件上滑的四種解法（T-pose、SMPL-X 中性；藍＝現在，橘＝解法；紅點＝手腕，虛線＝縫合前下襬）", fontsize=17)
    plt.tight_layout()
    plt.savefig(DST / "上滑解法_1_整件.png", dpi=60)
    plt.close(fig)

    # 圖二：數字
    keys = [("袖口離手腕cm", "袖口離手腕（cm，越接近 0 越好）"), ("下襬上移cm", "下襬比縫合前高（cm，越接近 0 越好）"),
            ("領口拉長%", "領口拉長（%，越小越好）"), ("穿進身體%", "穿進身體（%，越小越好）")]
    labels = [label for label, _ in METHODS.values()]
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
    width = 0.26
    colors = ["#5b8fd1", "#e8833a", "#59a14f"]
    for ax, (k, title) in zip(axes, keys):
        for i, g in enumerate(GARMENTS):
            vals = [res[g][label][k] for label in labels]
            bars = ax.bar(np.arange(len(labels)) + (i - 1) * width, vals, width, label=g.split("_")[0], color=colors[i])
            ax.bar_label(bars, fmt="%.1f", fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(labels)), [s.replace("＋", "\n＋") for s in labels], fontsize=11)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=12)
    fig.suptitle("四種解法的數字比較", fontsize=17)
    plt.tight_layout()
    plt.savefig(DST / "上滑解法_2_數字.png", dpi=65)
    print("ok")


if __name__ == "__main__":
    main()
