# -*- coding: utf-8 -*-
"""
女性模板身體（T-pose）換成新格式尺寸檔，袖子會不會跟 SMPL-X 中性一樣？

身體網格不變（f_smpl_template），尺寸檔換成：
  mean_all    （新格式、中性平均，172 cm）
  mean_female （新格式、女性平均，166 cm，和女性模板身高一樣）
手臂角度都改成 0（T-pose）。同 sleeve_smplx_test 的 g03、g05（沒縮口），新縫合。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_female_new_yaml.py
輸出：output/sleeve_female_yaml/<尺寸檔>_<衣服>_sewn.obj、results.json
"""
import json
import pickle
import sys
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
from export_for_physics import OUT as EXPORT, prepare_tpose_body  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402
from sleeve_sew_ablation import measure  # noqa: E402
from sleeve_smplx_test import NEW  # noqa: E402
from sleeve_tpose_test import arm_smpl  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "output" / "sleeve_female_yaml"
GARMENTS = ["aibf16_3090_v2_g03_blue_hoodie", "aibf16_3090_v2_g05_beige_longsleeve_tee"]
YAMLS = ["mean_all", "mean_female"]


def tpose_yaml(src):
    body = yaml.safe_load((dm.GC_ROOT / "assets" / "bodies" / f"{src}.yaml").read_text())
    body["body"]["arm_pose_angle"] = 0
    path = EXPORT / "_bodies" / f"tpose_{src}.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dm.ensure_system_json()
    body_obj, _ = prepare_tpose_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    arms = {}
    for side in ("left", "right"):
        c, u, w = arm_smpl(bv, side)
        arms[1 if c[0] > 0 else -1] = (c, u, w)
    results = {}
    for src in YAMLS:
        body_yaml = tpose_yaml(src)
        for g in GARMENTS:
            short = g.split("_v2_")[1]
            design = yaml.safe_load((dm.OUT_ROOT / g / "design_used.yaml").read_text(encoding="utf-8"))["design"]
            design["sleeve"]["cuff"]["type"]["v"] = None
            fix_neck_depth(design, BodyParameters(str(body_yaml)), MetaGarment)
            name = f"female_{src}_{short}"
            _, _, paths = dm.generate(design, body_yaml, name)
            folder = paths.g_box_mesh.parent
            v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
            E = mesh_edges(f)
            rest = rest_lengths(v, E, pickle.loads(paths.g_orig_edge_len.read_bytes()))
            seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
            sleeve = np.array(["sleeve" in s for s in seg])
            labels = yaml.safe_load((folder / f"{name}_vertex_labels.yaml").read_text())
            collar = np.zeros(len(v), bool)
            collar[[i for k, ids in labels.items() if "collar" in k for i in ids]] = True
            collar_e = collar[E[:, 0]] & collar[E[:, 1]]
            before = measure(v, sleeve, collar_e, rest, E, arms)
            x = sew(v, E, rest, body=(bv, bf), **NEW)
            igl.write_triangle_mesh(str(OUT / f"{src}_{short}_sewn.obj"), x, f)
            after = measure(x, sleeve, collar_e, rest, E, arms)
            results[f"{src} {short}"] = {"縫合前": before, "新縫合": after}
            print(f"{src} {short}　縫合前 {before}　新縫合 {after}", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
