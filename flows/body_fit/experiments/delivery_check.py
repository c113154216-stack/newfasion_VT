# -*- coding: utf-8 -*-
"""
交付前檢查（2026-11-09，回應紀泓宇 11-09 的回覆）：匯出的每件衣服量
  - 尖角三角形（和相鄰面夾角超過 120°，sew.folded_faces）
  - 自我穿插的面（CGAL remesh_self_intersections，只偵測）
  - 穿進身體的頂點 %（winding number）、胸前（肩關節到肚臍、身體前方）穿進身體的頂點數
  - 袖管鬆度：袖子中段（肩關節往外 30%～85% 手臂長）頂點到身體表面的距離中位數（mm），紀泓宇要約 18 mm
  - 頂點數（上限 4,000）

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python flows\\body_fit\\experiments\\delivery_check.py [匯出資料夾，預設 output/physics_export/neutral]
輸出：<匯出資料夾>/_logs/delivery_check.json，並印出摘要
"""
import json
import sys
from pathlib import Path

import igl
import igl.copyleft.cgal as cgal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sleevelen  # noqa: E402
import smplx_body  # noqa: E402
from sew import folded_faces  # noqa: E402

ROOT = HERE.parents[2]
BODY = ROOT / "output" / "physics_export" / "_bodies" / "smplx_neutral_theta0.obj"


def main(folder=None):
    folder = Path(folder) if folder else ROOT / "output" / "physics_export" / "neutral"
    bv, bf = igl.read_triangle_mesh(str(BODY))
    bv *= 100
    arm = sleevelen.arms(0.0)
    J = smplx_body.load_model("neutral")["J_regressor"] @ smplx_body.make_body("neutral")[0] * 100
    chest_y = (J[0, 1] + 5, J[12, 1])  # 骨盆往上 5 cm 到脖子
    out = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("_")):
        g = d.name
        v, f = igl.read_triangle_mesh(str(d / f"{g}.obj"))
        v *= 100
        _, _, IF, _, _ = cgal.remesh_self_intersections(v, f, detect_only=True)
        wn = igl.fast_winding_number_for_meshes(bv, bf, v)
        inside = wn > 0.5
        chest = inside & (v[:, 1] > chest_y[0]) & (v[:, 1] < chest_y[1]) & (v[:, 2] > J[9, 2]) & (np.abs(v[:, 0]) < 15)
        d2, _, _ = igl.point_mesh_squared_distance(v, bv, bf)
        dist = np.sqrt(d2) * 10  # mm
        ease = []
        for sign, (c, u, wrist) in arm.items():
            t = (v - c) @ u / wrist
            m = (sign * v[:, 0] > 0) & (t > 0.3) & (t < 0.85)
            if m.sum() > 20:
                ease.append(np.median(dist[m]))
        r = {"頂點": len(v), "尖角面": int(len(folded_faces(v, f))), "自我穿插面對": int(len(IF)),
             "穿進身體%": round(100 * float(inside.mean()), 2), "胸前穿進身體頂點": int(chest.sum()),
             "袖管鬆度 mm": round(float(np.mean(ease)), 1) if ease else None}
        out[g] = r
        print(g, r, flush=True)
    (folder / "_logs").mkdir(exist_ok=True)
    (folder / "_logs" / "delivery_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    vals = list(out.values())
    print("總計：頂點最多", max(x["頂點"] for x in vals), "、有尖角的件數", sum(x["尖角面"] > 0 for x in vals),
          "、有自我穿插的件數", sum(x["自我穿插面對"] > 0 for x in vals),
          "、袖管鬆度中位數", np.median([x["袖管鬆度 mm"] for x in vals if x["袖管鬆度 mm"]]))


if __name__ == "__main__":
    main(*sys.argv[1:])
