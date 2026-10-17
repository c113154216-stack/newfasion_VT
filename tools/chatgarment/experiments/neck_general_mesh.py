# -*- coding: utf-8 -*-
"""
用通用領深校正（neckline.py）＋新縫合設定，重新生成一批 AI 結果的網格，跟原本的並排比較。

跑法（在 newfashion_VT 根目錄，路徑不能有中文）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\neck_general_mesh.py results\\chatgarment_ai\\bf16_3090 aibf16_3090_neck [身體]

輸出：output/chatgarment/<tag>_<衣服>/<tag>_<衣服>_sewn.obj，以及 neck_fix.json（校正前後的數值）
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
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions  # noqa: E402

SEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}


def main(results, tag, body_name="mean_male"):
    dm.ensure_system_json()
    body_yaml = dm.GC_ROOT / "assets" / "bodies" / f"{body_name}.yaml"
    body = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_yaml.with_suffix(".obj")))
    for src in sorted(Path(results).glob("*/design_upper.yaml")):
        g = src.parent.name
        design = yaml.safe_load(src.read_text(encoding="utf-8"))["design"]
        design["meta"]["bottom"]["v"] = None
        design["meta"]["wb"]["v"] = None
        info = fix_neck_depth(design, body, MetaGarment)
        name = f"{tag}_{g}"
        _, _, paths = dm.generate(design, body_yaml, name)
        v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
        edges = mesh_edges(f)
        rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
        x = sew(v, edges, rest, body=(bv * 100, bf), **SEW)
        out = paths.g_box_mesh.with_name(paths.g_box_mesh.name.replace("_boxmesh.obj", "_sewn.obj"))
        write_obj_with_new_positions(paths.g_box_mesh, out, x)
        (out.parent / "neck_fix.json").write_text(json.dumps({"領口校正": info, "縫合": SEW}, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        (out.parent / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True), encoding="utf-8")
        print(f"[{g}] {info}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
