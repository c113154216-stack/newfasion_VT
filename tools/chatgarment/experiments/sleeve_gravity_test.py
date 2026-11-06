# -*- coding: utf-8 -*-
"""
實驗二 A：新縫合（拉長限制＋彎曲剛度）再加重力，能不能把被往上推的衣服（尤其袖子）拉回來？

10-17 試過只加重力（沒有拉長限制），布會被往下拉長（領口 +44%）。現在有拉長限制，布不能變長，
重力應該只能讓整件往下滑。同實驗一（sleeve_sew_ablation.py）的 3 件衣服，重力 0／0.005／0.01／0.02 各縫一次。
量：袖口離手腕、領口拉長 %、下襬離地、穿進身體的頂點比例。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_gravity_test.py
輸出：output/sleeve_gravity/<代號>_<衣服>_sewn.obj、results.json；
畫圖：.venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_ablation_figure.py sleeve_gravity_test
"""
import json
import pickle
import sys
import time
from pathlib import Path

import igl
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sleeve_sew_ablation import BODIES, GARMENTS, ROOT, SRC, arm, measure  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402

OUT = ROOT / "output" / "sleeve_gravity"
NEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}
METHODS = {  # 名稱: (檔名代號, 縫合參數)
    "重力 0（現在）": ("g0", dict(NEW)),
    "重力 0.005": ("g005", dict(NEW, gravity=0.005)),
    "重力 0.01": ("g01", dict(NEW, gravity=0.01)),
    "重力 0.02": ("g02", dict(NEW, gravity=0.02)),
}


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
        results[short] = {}
        for name, (code, kw) in METHODS.items():
            t0 = time.time()
            x = sew(v, E, rest, body=(bv, bf), **kw)
            igl.write_triangle_mesh(str(OUT / f"{code}_{short}_sewn.obj"), x, f)
            m = measure(x, sleeve, collar_e, rest, E, arms)
            m["穿進身體%"] = round(100 * float(np.mean(igl.signed_distance(x, bv, bf)[0] < 0)), 2)
            results[short][name] = m
            print(f"{short:28s} {name:10s} {m}（{time.time() - t0:.0f} 秒）", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
