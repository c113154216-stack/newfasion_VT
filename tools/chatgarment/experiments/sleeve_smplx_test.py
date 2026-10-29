# -*- coding: utf-8 -*-
"""
換成團隊約定的身體（SMPL-X 中性 θ=0、尺寸檔 mean_all＋手臂角度 0）之後，T-pose 的袖子到哪裡？

同實驗一的 3 件（design_used.yaml，領深照新身體重算），各做「沒縮口」「加縮口（CuffBand、收皺 1.7）」兩版，
新縫合縫一次，量：袖口離手腕、領口拉長 %、下襬離地、穿進身體 %。
手腕用 SMPL-X 的關節（J_regressor 的 20、21 號＝左右手腕；16、17＝左右肩），T-pose 手臂沿 ±x。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_smplx_test.py
輸出：output/sleeve_smplx/<版本>_<衣服>_sewn.obj、results.json
"""
import copy
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
import smplx_body  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402
from sleeve_sew_ablation import GARMENTS, measure  # noqa: E402
from to_design import CUFF_RUFFLE  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "output" / "sleeve_smplx"
NEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}


def arms_from_joints(lift):
    """{x 正負號: (肩關節, 手臂方向, 手腕在方向上的位置)}，單位 cm、腳底在 0。"""
    model = smplx_body.load_model("neutral")
    v, _, _ = smplx_body.make_body("neutral")
    J = (model["J_regressor"] @ v + [0.0, lift, 0.0]) * 100
    arms = {}
    for shoulder, wrist in ((16, 20), (17, 21)):
        c = J[shoulder]
        u = (J[wrist] - c) / np.linalg.norm(J[wrist] - c)
        arms[1 if J[wrist, 0] > 0 else -1] = (c, u, float((J[wrist] - c) @ u))
    return arms


def with_cuff(design):
    design = copy.deepcopy(design)
    design["sleeve"]["cuff"]["type"]["v"] = "CuffBand"
    design["sleeve"]["cuff"]["top_ruffle"]["v"] = CUFF_RUFFLE
    return design


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    arms = arms_from_joints(lift)
    print("手腕（肩→手腕 cm）:", {k: round(a[2], 1) for k, a in arms.items()})
    results = {}
    for g in GARMENTS:
        short = g.split("_v2_")[1]
        base = yaml.safe_load((dm.OUT_ROOT / g / "design_used.yaml").read_text(encoding="utf-8"))["design"]
        base["sleeve"]["cuff"]["type"]["v"] = None
        results[short] = {}
        for code, design in (("nocuff", base), ("cuff", with_cuff(base))):
            fix_neck_depth(design, BodyParameters(str(body_yaml)), MetaGarment)  # 領深照新身體重算
            name = f"smplx_{code}_{short}"
            _, _, paths = dm.generate(design, body_yaml, name)
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
            t0 = time.time()
            x = sew(v, E, rest, body=(bv, bf), **NEW)
            igl.write_triangle_mesh(str(OUT / f"{code}_{short}_sewn.obj"), x, f)
            after = measure(x, sleeve, collar_e, rest, E, arms)
            after["穿進身體%"] = round(100 * float(np.mean(igl.signed_distance(x, bv, bf)[0] < 0)), 2)
            results[short][code] = {"縫合前": before, "新縫合": after}
            print(f"{short} {code:6s} 縫合前 {before}\n{'':14s}新縫合 {after}（{time.time() - t0:.0f} 秒）", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
