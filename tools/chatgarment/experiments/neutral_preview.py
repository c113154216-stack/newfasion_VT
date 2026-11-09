# -*- coding: utf-8 -*-
r"""meshes/neutral/ 的總覽圖：每件正面（身體灰、衣服藍），標題是名稱和頂點數。

跑法（在 newfashion_VT 根目錄）：
    .venv\Scripts\python tools\chatgarment\experiments\neutral_preview.py
輸出：output/physics_export/neutral/preview.png
"""
import sys
from pathlib import Path

import igl
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from body_compare_figure import SKIN, shade, plt  # noqa: E402

ROOT = HERE.parents[2]
SRC = ROOT / "output" / "physics_export" / "neutral"
BODY = ROOT / "output" / "physics_export" / "_bodies" / "smplx_neutral_theta0.obj"
COLS = 8


def main():
    bv, bf = igl.read_triangle_mesh(str(BODY))
    bv *= 100
    keep = bv[bf].mean(1)[:, 1] > -45
    bf = bf[keep]
    items = sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("_"))
    rows = int(np.ceil(len(items) / COLS))
    fig, axes = plt.subplots(rows, COLS, figsize=(3.2 * COLS, 2.6 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, d in zip(axes.ravel(), items):
        v, f = igl.read_triangle_mesh(str(d / f"{d.name}.obj"))
        shade(ax, [(bv, bf, SKIN), (v * 100, f, "#5b8fd1")])
        ax.set_xlim(-95, 95), ax.set_ylim(-50, 40)
        ax.set_title(f"{d.name}\n{len(v):,} 頂點", fontsize=9)
    fig.suptitle(f"meshes/neutral/：{len(items)} 件，SMPL-X 中性、θ=0、β 全 0（2026-11-09：袖長規則＋尖角攤平）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(SRC / "preview.png", dpi=55)
    print("ok")


if __name__ == "__main__":
    main()
