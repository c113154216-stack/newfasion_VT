# -*- coding: utf-8 -*-
r"""幫已經匯出的帽 T 補上 `hood` 固定點組（帽子戴著用，規則同 export_neutral.hood_pins），不用整件重縫。
跑法（在 newfashion_VT 根目錄）：
    .venv\Scripts\python tools\chatgarment\experiments\add_hood_pins.py [匯出資料夾，預設 output/physics_export/neutral]
"""
import json
import sys
from pathlib import Path

import igl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import smplx_body  # noqa: E402
from export_neutral import hood_pins  # noqa: E402

ROOT = HERE.parents[2]
SEG = ROOT / "output" / "chatgarment"


def main(folder=None):
    folder = Path(folder) if folder else ROOT / "output" / "physics_export" / "neutral"
    bv, bf, _ = smplx_body.make_body("neutral")
    for d in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("_")):
        g = d.name
        seg = SEG / f"nx_{g}" / f"nx_{g}_sim_segmentation.txt"
        if not seg.exists():
            continue
        x, _ = igl.read_triangle_mesh(str(d / f"{g}.obj"))
        hood = hood_pins(x, seg.read_text().splitlines(), (bv, bf))
        if not hood:
            continue
        pj = d / f"{g}_pins.json"
        data = json.loads(pj.read_text(encoding="utf-8"))
        data["pins"]["hood"] = hood
        pj.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(g, "hood", len(hood), "neck_shoulder_band", len(data["pins"]["neck_shoulder_band"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
