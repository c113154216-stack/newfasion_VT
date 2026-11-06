# -*- coding: utf-8 -*-
"""
實驗三：新縫合再加「壓短限制」（compress_limit），袖子還會不會被往上推、擠短？

身體：SMPL-X 中性 T-pose（尺寸檔 mean_all＋手臂角度 0），同 sleeve_smplx_test 的 g03、g05（沒縮口）。
縫法：現在的新縫合／再加壓短限制 5%／再加壓短限制 2%。
量：袖口離手腕、領口拉長 %、下襬離地、穿進身體 %、袖子被壓到原長 9 成以下的邊 %。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_compress_test.py
加縮口版（CuffBand、收皺 1.7，只跑 CUFF_METHODS，結果併進同一份 results.json）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_compress_test.py cuff
輸出：output/sleeve_compress/<縫法代號>_<衣服>_sewn.obj、results.json
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
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402
from sleeve_sew_ablation import measure  # noqa: E402
from sleeve_smplx_test import NEW, arms_from_joints, with_cuff  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "output" / "sleeve_compress"
GARMENTS = ["aibf16_3090_v2_g03_blue_hoodie", "aibf16_3090_v2_g05_beige_longsleeve_tee"]
METHODS = {
    "新縫合（現在）": ("new", NEW),
    "＋壓短限制 5%": ("c05", {**NEW, "compress_limit": 0.05}),
    "＋壓短限制 2%": ("c02", {**NEW, "compress_limit": 0.02}),
}
CUFF_METHODS = {
    "縮口＋壓短限制 2%": ("cuffc02", {**NEW, "compress_limit": 0.02}),
}


def main(mode="nocuff"):
    cuff = mode == "cuff"
    methods = CUFF_METHODS if cuff else METHODS
    OUT.mkdir(parents=True, exist_ok=True)
    dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    arms = arms_from_joints(lift)
    path = OUT / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for g in GARMENTS:
        short = g.split("_v2_")[1]
        design = yaml.safe_load((dm.OUT_ROOT / g / "design_used.yaml").read_text(encoding="utf-8"))["design"]
        design = with_cuff(design) if cuff else design
        if not cuff:
            design["sleeve"]["cuff"]["type"]["v"] = None
        fix_neck_depth(design, BodyParameters(str(body_yaml)), MetaGarment)
        name = f"smplx_{mode}_{short}"
        _, _, paths = dm.generate(design, body_yaml, name)
        folder = paths.g_box_mesh.parent
        v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
        E = mesh_edges(f)
        rest = rest_lengths(v, E, pickle.loads(paths.g_orig_edge_len.read_bytes()))
        seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
        sleeve = np.array(["sleeve" in s or "cuff" in s for s in seg])
        sleeve_e = sleeve[E[:, 0]] & sleeve[E[:, 1]]
        labels = yaml.safe_load((folder / f"{name}_vertex_labels.yaml").read_text())
        collar = np.zeros(len(v), bool)
        collar[[i for k, ids in labels.items() if "collar" in k for i in ids]] = True
        collar_e = collar[E[:, 0]] & collar[E[:, 1]]
        before = measure(v, sleeve, collar_e, rest, E, arms)
        results.setdefault(short, {})["縫合前" + ("（縮口）" if cuff else "")] = before
        print(f"\n{short}　縫合前 {before}", flush=True)
        for label, (code, kw) in methods.items():
            t0 = time.time()
            x = sew(v, E, rest, body=(bv, bf), **kw)
            igl.write_triangle_mesh(str(OUT / f"{code}_{short}_sewn.obj"), x, f)
            m = measure(x, sleeve, collar_e, rest, E, arms)
            m["穿進身體%"] = round(100 * float(np.mean(igl.signed_distance(x, bv, bf)[0] < 0)), 2)
            ratio = np.linalg.norm(x[E[sleeve_e, 0]] - x[E[sleeve_e, 1]], axis=1) / rest[sleeve_e]
            m["袖子被壓短邊%"] = round(100 * float(np.mean(ratio < 0.9)), 1)
            results[short][label] = m
            print(f"  {label:10s} {m}（{time.time() - t0:.0f} 秒）", flush=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
