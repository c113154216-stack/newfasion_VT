# -*- coding: utf-8 -*-
"""
實驗三（sleeve_compress_test.py）的對比圖：每列一件衣服，每欄一種縫法，紅點＝手腕關節。

跑法（在 newfashion_VT 根目錄，先跑過 sleeve_compress_test.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_compress_figure.py
輸出：output/chatgarment/袖長_壓短限制對比.png
"""
import json
import sys
from pathlib import Path

import igl
import matplotlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from body_compare_figure import BODIES, NEUTRAL, SKIN, load, shade  # noqa: E402
from sleeve_compress_test import CUFF_METHODS, GARMENTS, METHODS, OUT  # noqa: E402
from sleeve_smplx_test import arms_from_joints  # noqa: E402

DST = HERE.parents[2] / "output" / "chatgarment" / "袖長_壓短限制對比.png"


def main():
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    methods = {k: v for k, v in {**METHODS, **CUFF_METHODS}.items() if k in next(iter(results.values()))}
    body, bf = load(BODIES / "smplx_neutral_theta0_feet0.obj", 100)
    lift = -igl.read_triangle_mesh(str(BODIES / "smplx_neutral_theta0.obj"))[0][:, 1].min()
    wrists = [c + u * w for c, u, w in arms_from_joints(lift).values()]
    fig, axes = plt.subplots(len(GARMENTS), len(methods), figsize=(7 * len(methods), 5 * len(GARMENTS)))
    for r, g in enumerate(GARMENTS):
        short = g.split("_v2_")[1]
        for c, (label, (code, _)) in enumerate(methods.items()):
            ax = axes[r, c]
            gv, gf = load(OUT / f"{code}_{short}_sewn.obj", 1)
            shade(ax, [(body, bf, SKIN), (gv, gf, NEUTRAL)])
            for p in wrists:
                ax.scatter(p[0], p[1], s=60, c="red", zorder=5)
            ax.axhline(results[short]["縫合前"]["下襬離地cm"], color="k", ls="--", lw=1)
            ax.set_xlim(-92, 92); ax.set_ylim(75, 180); ax.grid(alpha=0.25)
            m = results[short][label]
            ax.set_title(f"{short}　{label}\n袖口離手腕 {m['袖口離手腕cm']:+.1f} cm　下襬 {m['下襬離地cm']:.0f} cm"
                         f"　袖子被壓短邊 {m['袖子被壓短邊%']:.0f}%", fontsize=12)
    fig.suptitle("實驗三：加壓短限制（T-pose、SMPL-X 中性；紅點＝手腕，虛線＝縫合前的下襬）", fontsize=16)
    plt.tight_layout()
    plt.savefig(DST, dpi=70)
    print(DST)


if __name__ == "__main__":
    main()
