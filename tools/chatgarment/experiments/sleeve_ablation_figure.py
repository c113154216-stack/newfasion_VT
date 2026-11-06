# -*- coding: utf-8 -*-
"""
把實驗一（sleeve_sew_ablation.py）縫好的網格畫成對比圖：每列一件衣服，每欄一種縫法。
紅點＝手腕位置；標題是袖口離手腕的距離和下襬高度。

跑法（在 newfashion_VT 根目錄，先跑過 sleeve_sew_ablation.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_ablation_figure.py [實驗模組]
實驗模組預設 sleeve_sew_ablation（實驗一）；sleeve_gravity_test 是實驗二 A（重力）。
輸出：output/chatgarment/袖長_縫法對比.png（實驗二 A：袖長_重力對比.png）
"""
import json
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gallery import draw, panel_colors, shaded  # noqa: E402
from sleeve_sew_ablation import BODIES, GARMENTS, SRC, arm  # noqa: E402

FIGURES = {"sleeve_sew_ablation": ("袖長_縫法對比.png", "長袖縫完變短：4 種縫法對比"),
           "sleeve_gravity_test": ("袖長_重力對比.png", "新縫合加重力（實驗二 A）")}

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


def main(module="sleeve_sew_ablation"):
    exp = __import__(module)
    OUT, METHODS = exp.OUT, exp.METHODS
    filename, heading = FIGURES[module]
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    bv, bf = igl.read_triangle_mesh(str(BODIES / "mean_male.obj"))
    bv *= 100
    bf = bf[bv[bf].mean(axis=1)[:, 1] > 60]  # 身體只畫膝蓋以上
    body_color = shaded(bv, bf, np.array([0.80, 0.66, 0.54]))
    wrists = [c + u * w for c, u, w in (arm(bv, s) for s in (1, -1))]
    first = next(iter(results.values()))
    cols = [("縫合前", None)] * ("縫合前" in first) + [(name, code) for name, (code, _) in METHODS.items()]

    fig = plt.figure(figsize=(4 * len(cols), 4.6 * len(GARMENTS)))
    for r, g in enumerate(GARMENTS):
        short = g.split("_v2_")[1]
        box = SRC / g / f"{g}_boxmesh.obj"
        for k, (name, code) in enumerate(cols):
            ax = fig.add_subplot(len(GARMENTS), len(cols), r * len(cols) + k + 1, projection="3d")
            gv, gf = igl.read_triangle_mesh(str(box if code is None else OUT / f"{code}_{short}_sewn.obj"))
            draw(ax, [(bv, bf, body_color), (gv, gf, shaded(gv, gf, panel_colors(SRC / g, g, gf)))])
            ax.computed_zorder = False  # 照加入順序畫，紅點才不會被衣服蓋住
            for p in wrists:
                ax.scatter(p[0], p[2], p[1], s=60, c="red", depthshade=False)
            ax.set_xlim(-55, 55); ax.set_ylim(-55, 55); ax.set_zlim(65, 175)
            ax.view_init(8, 90); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
            m = results[short][name]
            ax.set_title(f"{name}\n袖口離手腕 {m['袖口離手腕cm']:+.1f} cm　下襬 {m['下襬離地cm']:.0f} cm", fontsize=12)
        fig.text(0.005, 1 - (r + 0.5) / len(GARMENTS), short, rotation=90, va="center", fontsize=13)
    fig.suptitle(f"{heading}（紅點＝手腕）", fontsize=16)
    plt.tight_layout(rect=(0.015, 0, 1, 0.97))
    dst = SRC / filename
    plt.savefig(dst, dpi=70)
    print(dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
