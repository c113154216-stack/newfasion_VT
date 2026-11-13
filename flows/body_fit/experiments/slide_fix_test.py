# -*- coding: utf-8 -*-
"""
實驗四：整件衣服往上滑，四種解法比較（重力、放鬆領口、拉回原高度、放鬆領口＋拉回原高度）。

身體：SMPL-X 中性 T-pose（mean_all 尺寸檔）。衣服：交付版（export_neutral.py）的款式，g03 藍帽 T、g05 米色長袖、h09 高領毛衣，
頂點間距 2 cm（和交付相同）。所有縫法都以現在的交付縫法（新縫合＋壓短限制 2%）為基礎，只多加一項。
量：袖口離手腕、下襬離地、下襬比縫合前高多少、領口拉長 %、穿進身體 %。

跑法（在 newfashion_VT 根目錄，可以每件各開一個程序同時跑）：
    .venv\\Scripts\\python flows\\body_fit\\experiments\\slide_fix_test.py [衣服…]
輸出：output/slide_fix/<縫法代號>_<衣服>_sewn.obj、results_<衣服>.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import igl
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import RESOLUTION, SEW  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402
from sleeve_sew_ablation import measure  # noqa: E402
from sleeve_smplx_test import arms_from_joints  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "output" / "slide_fix"
SRC = ROOT / "output" / "physics_export" / "neutral"
GARMENTS = ["g03_blue_hoodie", "g05_beige_longsleeve_tee", "h09_navy_turtleneck_sweater"]
NECK_RELAX = 0.10  # 領口那一圈可以拉長 10%（其他邊維持 2%）
# 代號 → (名稱, 多加的參數)；"neck" 表示領口那圈的拉長限制放鬆
METHODS = {
    "now": ("現在（對照）", {}),
    "g02": ("重力 0.02", {"gravity": 0.02}),
    "g05": ("重力 0.05", {"gravity": 0.05}),
    "neck": ("放鬆領口 10%", {"neck": True}),
    "ay": ("拉回原高度", {"anchor_y": 0.02}),
    "neck_ay": ("放鬆領口＋拉回原高度", {"neck": True, "anchor_y": 0.02}),
}


def main(*only):
    OUT.mkdir(parents=True, exist_ok=True)
    dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    arms = arms_from_joints(lift)
    for g in only or GARMENTS:
        design = yaml.safe_load((SRC / g / "design_used.yaml").read_text(encoding="utf-8"))["design"]
        name = f"slide_{g}"
        _, _, paths = dm.generate(design, body_yaml, name, resolution=RESOLUTION)
        folder = paths.g_box_mesh.parent
        v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
        E = mesh_edges(f)
        rest = rest_lengths(v, E, pickle.loads(paths.g_orig_edge_len.read_bytes()))
        seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
        sleeve = np.array(["sleeve" in s or "cuff" in s for s in seg])
        labels = yaml.safe_load((folder / f"{name}_vertex_labels.yaml").read_text())
        collar = np.zeros(len(v), bool)
        collar[[i for k, ids in labels.items() if "collar" in k for i in ids]] = True
        collar_e = collar[E[:, 0]] & collar[E[:, 1]]
        before = measure(v, sleeve, collar_e, rest, E, arms)
        results = {"縫合前": before}
        print(f"\n{g}　縫合前 {before}", flush=True)
        for code, (label, extra) in METHODS.items():
            t0 = time.time()
            kw = {**SEW, **{k: val for k, val in extra.items() if k != "neck"}}
            if extra.get("neck"):
                kw["stretch_limit"] = np.where(collar_e, NECK_RELAX, SEW["stretch_limit"])
            x = sew(v, E, rest, body=(bv, bf), **kw)
            igl.write_triangle_mesh(str(OUT / f"{code}_{g}_sewn.obj"), x, f)
            m = measure(x, sleeve, collar_e, rest, E, arms)
            m["下襬上移cm"] = round(m["下襬離地cm"] - before["下襬離地cm"], 1)
            m["穿進身體%"] = round(100 * float(np.mean(igl.fast_winding_number_for_meshes(bv, bf, x) > 0.5)), 2)
            results[label] = m
            print(f"  {label:12s} {m}（{time.time() - t0:.0f} 秒）", flush=True)
        (OUT / f"results_{g}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
