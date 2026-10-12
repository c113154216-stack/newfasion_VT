# -*- coding: utf-8 -*-
"""把 export_for_physics.py 的輸出畫成一張檢查圖：每件衣服的正面與側面，紅點＝領口固定點、橘點＝肩線固定點。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\physics_preview.py

輸出：output/physics_export/preview.png
"""
import json
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

OUT = Path(__file__).resolve().parents[2] / "output" / "physics_export"


def main():
    items = sorted(p for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_"))
    fig = plt.figure(figsize=(9, 4.3 * len(items)))
    for k, folder in enumerate(items):
        info = json.loads((folder / f"{folder.name}_pins.json").read_text(encoding="utf-8"))
        bv, bf = igl.read_triangle_mesh(str(OUT.parent.parent / info["body_obj"]))
        gv, gf = igl.read_triangle_mesh(str(folder / f"{folder.name}.obj"))
        bv, gv = bv * 100, gv * 100  # 畫圖用公分
        bf = bf[bv[bf].mean(axis=1)[:, 1] > 70]
        garment_color = shaded(gv, gf, np.tile([[0.42, 0.5, 0.84]], (len(gf), 1)))
        for j, (azim, view) in enumerate([(90, "正面"), (0, "側面")]):
            ax = fig.add_subplot(len(items), 2, 2 * k + j + 1, projection="3d")
            draw(ax, [(bv, bf, shaded(bv, bf, np.array([0.80, 0.66, 0.54]))), (gv, gf, garment_color)])
            for key, color in (("neckline", "red"), ("shoulder", "orange")):
                p = gv[info["pins"][key]]
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=color, s=8, depthshade=False)
            ax.set_xlim(-85, 85); ax.set_ylim(-85, 85); ax.set_zlim(60, 175)
            ax.view_init(5, azim); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
            ax.set_title(f"{folder.name}・{view}（{info['n_vertices']:,} 頂點）", fontsize=11)
    fig.suptitle("交給物理模擬的衣服（T-pose、公尺）。紅＝領口固定點，橘＝肩線固定點", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    path = OUT / "preview.png"
    plt.savefig(path, dpi=70)
    print(path)


if __name__ == "__main__":
    main()
