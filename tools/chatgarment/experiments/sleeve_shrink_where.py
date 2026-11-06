# -*- coding: utf-8 -*-
"""
實驗六：(1) 版片初始位置往上移，縫好的結果會不會變；(2) 袖子是在哪一段變短的。

g05 米色長袖，交付縫法（新縫合＋壓短限制 2%），SMPL-X 中性 T-pose。
袖子沿手臂方向（肩關節 → 手腕）量三個位置：袖子內端（接肩膀那一圈，取最近的 2%）、袖口（最遠）、兩者之差（袖子本身多長）。
跑法（在 newfashion_VT 根目錄，先跑過 where_push.py）：
    .venv\Scripts\python tools\chatgarment\experiments\sleeve_shrink_where.py
輸出：output/slide_fix/shrink_<代號>_g05_sewn.obj、shrink_g05.json
"""
import json
import pickle
import sys
from pathlib import Path

import igl
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sleevelen  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import SEW  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402

ROOT = HERE.parents[2]
G = "g05_beige_longsleeve_tee"
SRC = ROOT / "output" / "chatgarment" / f"slide_{G}" / f"slide_{G}"
OUT = ROOT / "output" / "slide_fix"


def sleeve_stats(x, sleeve, arm):
    inner, outer = [], []
    for sign, (c, u, wrist) in arm.items():
        t = (x[sleeve & (sign * x[:, 0] > 0)] - c) @ u
        inner.append(np.percentile(t, 2))
        outer.append(t.max() - wrist)
    i, o = float(np.mean(inner)), float(np.mean(outer))
    return {"袖子內端（離肩關節 cm）": round(i, 1), "袖口離手腕 cm": round(o, 1)}


def main():
    body_obj, _, lift = prepare_smplx_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    arm = sleevelen.arms(lift)
    v, f = igl.read_triangle_mesh(str(SRC) + "_boxmesh.obj")
    E = mesh_edges(f)
    rest = rest_lengths(v, E, pickle.loads(Path(str(SRC) + "_orig_lens.pickle").read_bytes()))
    sleeve = sleevelen.sleeve_mask(Path(str(SRC) + "_sim_segmentation.txt").read_text().splitlines())
    wrist = np.mean([w for _, _, w in arm.values()])
    res = {"手腕離肩關節 cm": round(float(wrist), 1), "縫合前": {**sleeve_stats(v, sleeve, arm), "下襬": round(float(v[:, 1].min()), 1)}}
    for code, label, v0, body in [("now", "縫合後（有身體）", v, (bv, bf)), ("nobody", "縫合後（不放身體）", v, None),
                                  ("up6", "版片先往上移 6 cm 再縫", v + [0, 6, 0], (bv, bf))]:
        x = sew(v0, E, rest, body=body, **SEW)
        igl.write_triangle_mesh(str(OUT / f"shrink_{code}_g05_sewn.obj"), x, f)
        res[label] = {**sleeve_stats(x, sleeve, arm), "下襬": round(float(x[:, 1].min()), 1)}
        print(label, res[label], flush=True)
    for k, r in res.items():
        if isinstance(r, dict):
            r["袖子本身長 cm"] = round(r["袖口離手腕 cm"] + wrist - r["袖子內端（離肩關節 cm）"], 1)
    (OUT / "shrink_g05.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
