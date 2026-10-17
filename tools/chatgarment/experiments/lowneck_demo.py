# -*- coding: utf-8 -*-
"""
低領衣服經過領口校正（neckline.py）會變怎樣：同一件衣服做幾種低領，各生成「原本」與「校正後」，
縫合都用新設定，差別只來自領口校正。

跑法（在 newfashion_VT 根目錄，路徑不能有中文）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\lowneck_demo.py <底的 design_upper.yaml>
"""
import copy
import json
import pickle
import sys
from pathlib import Path

import igl
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import design_to_mesh as dm  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from neckline import FC_RANGE, fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions  # noqa: E402

SEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}
# 名稱 → (領口形狀, AI 的前領深 0～1)
VARIANTS = {
    "v_u002": ("VNeckHalf", 0.02),
    "v_u025": ("VNeckHalf", 0.25),
    "scoop_u030": ("CircleNeckHalf", 0.30),
}


def build(design, body_yaml, body_obj, name, info=None):
    _, _, paths = dm.generate(design, body_yaml, name)
    v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
    edges = mesh_edges(f)
    rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
    x = sew(v, edges, rest, body=body_obj, **SEW)
    write_obj_with_new_positions(paths.g_box_mesh, paths.g_box_mesh.with_name(f"{name}_sewn.obj"), x)
    (paths.g_box_mesh.parent / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True),
                                                              encoding="utf-8")
    if info:
        (paths.g_box_mesh.parent / "neck_fix.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main(base_yaml, body_name="mean_male"):
    dm.ensure_system_json()
    body_yaml = dm.GC_ROOT / "assets" / "bodies" / f"{body_name}.yaml"
    body = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_yaml.with_suffix(".obj")))
    base = yaml.safe_load(Path(base_yaml).read_text(encoding="utf-8"))["design"]
    base["meta"]["bottom"]["v"] = None
    base["meta"]["wb"]["v"] = None
    for g, (shape, u) in VARIANTS.items():
        d = copy.deepcopy(base)
        d["collar"]["f_collar"]["v"] = d["collar"]["b_collar"]["v"] = shape
        d["collar"]["fc_depth"]["v"] = FC_RANGE[0] + u * (FC_RANGE[1] - FC_RANGE[0])
        build(copy.deepcopy(d), body_yaml, (bv * 100, bf), f"lowneck_orig_{g}")
        info = fix_neck_depth(d, body, MetaGarment)
        build(d, body_yaml, (bv * 100, bf), f"lowneck_fix_{g}", info)
        print(f"[{g}] {shape} u={u} → {info}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
