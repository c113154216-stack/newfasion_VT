# -*- coding: utf-8 -*-
r"""把 spec_batch.py 縫好的各尺碼並排出圖：商品照＋每個尺碼的正面、背面。

跑法（在 newfashion_VT 根目錄，先跑過 spec_batch.py）：
    .venv\Scripts\python flows\size_chart\spec_batch_figure.py [代號 ...]     # 不給代號就全部
輸出：output/spec_to_pattern/_batch/<代號>/尺碼並排.png
每格標題：縫好後沿布面量的 3D 數字／尺寸表（單位 cm），差超過 2 cm 的項目標 *
"""
import json
import sys
from pathlib import Path

import igl
import matplotlib.image as mpimg
import numpy as np

HERE = Path(__file__).resolve().parent
BODY_FIT = HERE.parent / "body_fit"
sys.path.insert(0, str(BODY_FIT))
sys.path.insert(0, str(BODY_FIT / "experiments"))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
from export_for_physics import prepare_smplx_body  # noqa: E402
from body_compare_figure import SKIN, shade, plt  # noqa: E402

ROOT = sp.ROOT
OUT = ROOT / "output" / "spec_to_pattern" / "_batch"
PHOTOS = HERE / "網路衣服_尺寸資料"
ORDER = ["XXS", "XS", "S", "M", "L", "XL", "2XL", "XXL"]
NAMES = [("chest", "胸寬"), ("shoulder", "肩寬"), ("length", "衣長"), ("sleeve", "袖長"), ("hem", "下襬圍")]


def label(rec):
    table, got = rec["尺寸表"], rec["縫好量到"]
    lines = []
    for k, zh in NAMES:
        if table.get(k) is None or got.get(k) is None:
            continue
        mark = " *" if abs(got[k] - table[k]) > 2 else ""
        lines.append(f"{zh} {got[k]:.1f}／{table[k]:g}{mark}")
    return "\n".join(lines)


def figure(gdir, body):
    sizes = sorted([d for d in gdir.iterdir() if (d / "量測.json").exists()],
                   key=lambda d: ORDER.index(d.name) if d.name in ORDER else 99)
    if not sizes:
        return None
    recs = [json.loads((d / "量測.json").read_text(encoding="utf-8")) for d in sizes]
    name = recs[0]["衣服"]
    meshes = [igl.read_triangle_mesh(str(d / f"{gdir.name}_{d.name}.obj")) for d in sizes]
    top = max(x[:, 1].max() for x, _ in meshes) + 5
    bottom = min(x[:, 1].min() for x, _ in meshes) - 5
    half = max(np.abs(x[:, 0]).max() for x, _ in meshes) + 5

    flip = np.array([-1.0, 1.0, -1.0])
    n = 1 + len(sizes)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 9.6))
    for r, side in enumerate(["front", "back"]):
        ax = axes[r, 0]
        ax.axis("off")
        img = PHOTOS / name / f"{name}_{side}.jpg"
        if img.exists():
            ax.imshow(mpimg.imread(str(img)))
        ax.set_title("商品照（正面）" if side == "front" else "商品照（背面）", fontsize=12)
        for c, ((x, f), d, rec) in enumerate(zip(meshes, sizes, recs), start=1):
            ax = axes[r, c]
            s = flip if side == "back" else 1.0
            shade(ax, [(body[0] * s, body[1], SKIN), (x * s, f, "#e8e8e8")])
            ax.set_xlim(-half, half), ax.set_ylim(bottom, top)
            ax.set_aspect("equal")
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{d.name} 號\n{label(rec)}", fontsize=10, loc="left")
            else:
                ax.set_title(f"{d.name} 號（背面）", fontsize=10)
    fig.suptitle(f"{name}：照尺寸表打版、縫好後的各尺碼（標題＝縫好量到／尺寸表，cm；* 差超過 2 cm）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = gdir / "尺碼並排.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png


def main(codes):
    body_obj, _, _ = prepare_smplx_body("neutral")
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    gdirs = [OUT / c for c in codes] if codes else sorted(d for d in OUT.iterdir() if d.is_dir() and d.name.startswith("z"))
    for gdir in gdirs:
        png = figure(gdir, (bv * 100, bf))
        print("輸出：", png, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
