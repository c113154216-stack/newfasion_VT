# -*- coding: utf-8 -*-
"""
實驗一：長袖縫完變短，是新縫合的哪個改動造成的？

新縫合比舊縫合多了兩個改動：① 拉長限制（超過 2% 權重變 100）、② 彎曲剛度（bend 1）。
同一件衣服（沿用 hem_mesh.py 生成好的 box mesh）用 4 種縫法各縫一次，量三件事：
  袖口離手腕幾 cm（負＝還沒到手腕，左右平均）、領口一圈被拉長幾 %、下襬高度（離地 cm）

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_sew_ablation.py
輸出：output/sleeve_ablation/<縫法代號>_<衣服>_sewn.obj、results.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import igl
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "chatgarment"))
from sew import mesh_edges, rest_lengths, sew  # noqa: E402

SRC = ROOT / "output" / "chatgarment"
OUT = ROOT / "output" / "sleeve_ablation"
BODIES = ROOT / "third_party" / "GarmentCodeRC" / "assets" / "bodies"
GARMENTS = ["aibf16_3090_v2_g03_blue_hoodie", "aibf16_3090_v2_g05_beige_longsleeve_tee",
            "aibf16_h_v2_h09_navy_turtleneck_sweater"]
METHODS = {  # 名稱: (檔名用的英文代號, 縫合參數)；igl 寫檔不支援中文路徑
    "舊": ("old", {}),
    "只加①拉長限制": ("stretch", {"stretch_limit": 0.02, "stretch_weight": 100.0}),
    "只加②彎曲剛度": ("bend", {"bend": 1.0}),
    "新（現在）": ("new", {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}),
}


def arm(bv, sign):
    """手臂軸（PCA）和手腕在軸上的位置：手掌前最細的地方。"""
    seg = json.loads((BODIES / "ggg_body_segmentation.json").read_text())
    a = bv[seg["left_arm"]]
    if a[:, 0].mean() * sign < 0:
        a = bv[seg["right_arm"]]
    c = a.mean(0)
    u = np.linalg.svd(a - c)[2][0]
    u = u if u[0] * sign > 0 else -u
    t = (a - c) @ u
    r = np.linalg.norm((a - c) - np.outer(t, u), axis=1)
    edges = np.linspace(t.min(), t.max(), 60)
    mid, rad = (edges[:-1] + edges[1:]) / 2, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        rad.append(np.percentile(r[m], 90) if m.sum() > 5 else np.inf)
    span = t.max() - t.min()
    ok = (mid > t.min() + 0.55 * span) & (mid < t.max() - 0.12 * span)
    return c, u, mid[ok][np.argmin(np.array(rad)[ok])]


def measure(x, sleeve, collar_e, rest, E, arms):
    cuff = []
    for sign, (c, u, wrist) in arms.items():
        side = sleeve & (sign * x[:, 0] > 0)
        cuff.append(((x[side] - c) @ u).max() - wrist)
    neck = np.linalg.norm(x[E[collar_e, 0]] - x[E[collar_e, 1]], axis=1).sum() / rest[collar_e].sum() - 1
    return {"袖口離手腕cm": round(float(np.mean(cuff)), 1), "領口拉長%": round(100 * float(neck), 1),
            "下襬離地cm": round(float(x[~sleeve, 1].min()), 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bv, bf = igl.read_triangle_mesh(str(BODIES / "mean_male.obj"))
    bv *= 100
    arms = {s: arm(bv, s) for s in (1, -1)}
    results = {}
    for g in GARMENTS:
        d = SRC / g
        v, f = igl.read_triangle_mesh(str(d / f"{g}_boxmesh.obj"))
        E = mesh_edges(f)
        rest = rest_lengths(v, E, pickle.loads((d / f"{g}_orig_lens.pickle").read_bytes()))
        seg = (d / f"{g}_sim_segmentation.txt").read_text().splitlines()
        sleeve = np.array(["sleeve" in s for s in seg])
        labels = yaml.safe_load((d / f"{g}_vertex_labels.yaml").read_text())
        collar = np.zeros(len(v), bool)
        collar[[i for k, ids in labels.items() if "collar" in k for i in ids]] = True
        collar_e = collar[E[:, 0]] & collar[E[:, 1]]
        short = g.split("_v2_")[1]
        results[short] = {"縫合前": measure(v, sleeve, collar_e, rest, E, arms)}
        print(f"\n{short}　縫合前 {results[short]['縫合前']}", flush=True)
        for name, (code, kw) in METHODS.items():
            t0 = time.time()
            x = sew(v, E, rest, body=(bv, bf), **kw)
            igl.write_triangle_mesh(str(OUT / f"{code}_{short}_sewn.obj"), x, f)
            results[short][name] = measure(x, sleeve, collar_e, rest, E, arms)
            print(f"  {name:10s} {results[short][name]}（{time.time() - t0:.0f} 秒）", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
