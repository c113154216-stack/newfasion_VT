# -*- coding: utf-8 -*-
"""
衣長校正（hemline.py）＋領口校正（neckline.py）＋新縫合，重新生成一批 AI 結果的網格。

需要先跑過 neck_general_mesh.py（同一批、tag 為 <base_tag>）：用它縫好的網格量「AI 原本衣長」縫好後的下襬。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\hem_mesh.py ^
        results\\chatgarment_ai\\bf16_3090 aibf16_3090_neck aibf16_3090_hem

每件：決定目標下襬 → 生成、縫合 → 量縫好的下襬，差超過 1 cm 就修正衣長再縫一次。
輸出：output/chatgarment/<tag>_<衣服>/（網格、hem_fix.json、design_used.yaml）
"""
import ast
import copy
import json
import pickle
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

CWD = Path.cwd()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from hemline import correct, plan  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions  # noqa: E402

SEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}
TOLERANCE_CM = 1.0


def description(result_dir):
    text = (result_dir / "output.txt").read_text(encoding="utf-8")
    desc = ast.literal_eval(text.split("[描述]")[1].split("[版型]")[0].strip())
    garment = desc.get("upperbody_garment") or desc.get("wholebody_garment") or {}
    return garment.get("geometry_styles", {}), garment.get("garment_name")


def sewn_hem(tag, g):
    v, _ = igl.read_triangle_mesh(str(dm.OUT_ROOT / f"{tag}_{g}" / f"{tag}_{g}_sewn.obj"))
    return float(v[:, 1].min())


def build(design, name, body_yaml, body_mesh):
    _, _, paths = dm.generate(design, body_yaml, name)
    v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
    edges = mesh_edges(f)
    rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
    x = sew(v, edges, rest, body=body_mesh, **SEW)
    out = paths.g_box_mesh.with_name(paths.g_box_mesh.name.replace("_boxmesh.obj", "_sewn.obj"))
    write_obj_with_new_positions(paths.g_box_mesh, out, x)
    return out.parent, float(x[:, 1].min())


def main(results, base_tag, tag, body_name="mean_male", *only):
    """only：只做這幾件（資料夾名稱），例如補跑；不給就全部。"""
    dm.ensure_system_json()
    body_yaml = dm.GC_ROOT / "assets" / "bodies" / f"{body_name}.yaml"
    body = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_yaml.with_suffix(".obj")))
    body_mesh = (bv * 100, bf)

    # 上衣存在 design_upper.yaml；連身款（例如背心被判成 wholebody）存在 design_wholebody.yaml
    sources = {}
    for part in ("wholebody", "upper"):
        for src in (CWD / results).glob(f"*/design_{part}.yaml"):
            sources[src.parent.name] = src
    items = []
    for _, src in sorted(sources.items()):
        design = yaml.safe_load(src.read_text(encoding="utf-8"))["design"]
        design["meta"]["bottom"]["v"] = None
        design["meta"]["wb"]["v"] = None
        items.append((src.parent, design))
    # 一般款先做：合身款改成一般款後沒有縫好的參考，借用一般款量到的「縫合往上提」
    items.sort(key=lambda t: t[1]["meta"]["upper"]["v"] == "FittedShirt")

    lifts = []
    for result_dir, design in items:
        g = result_dir.name
        styles, garment_name = description(result_dir)
        fitted = design["meta"]["upper"]["v"] == "FittedShirt"
        ai_hem = None if fitted else sewn_hem(base_tag, g)
        lift = float(np.median(lifts)) if fitted else None
        if only and g not in only:
            continue
        info = plan(design, body, MetaGarment, styles, garment_name, ai_sewn_hem=ai_hem, lift=lift)
        neck = fix_neck_depth(design, body, MetaGarment)
        name = f"{tag}_{g}"
        if info is None:  # 沒有衣長規則：衣長維持 AI 的，領口校正、新縫合照做
            folder, hem = build(design, name, body_yaml, body_mesh)
            (folder / "hem_fix.json").write_text(json.dumps(
                {"衣長校正": "沒有對應的規則，衣長不改", "縫好的下襬 cm": round(hem, 1), "領口校正": neck, "縫合": SEW},
                ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True),
                                                     encoding="utf-8")
            print(f"[{g}] 沒有對應的規則，衣長不改（下襬 {hem:.1f} cm）", flush=True)
            continue
        if not fitted:
            lifts.append(info["縫合往上提 cm"])

        folder, hem = build(design, name, body_yaml, body_mesh)
        passes = [round(hem, 1)]
        if abs(hem - info["_target"]) > TOLERANCE_CM:  # 縫合往上提的量會隨衣長稍微改變，修正一次
            correct(design, info, hem)
            folder, hem = build(design, name, body_yaml, body_mesh)
            passes.append(round(hem, 1))
        info["縫好的下襬 cm（每一輪）"] = passes
        info["最終衣長參數"] = round(float(design["shirt"]["length"]["v"]), 3)
        report = {k: v for k, v in info.items() if not k.startswith("_")}
        (folder / "hem_fix.json").write_text(
            json.dumps({"衣長校正": report, "領口校正": neck, "縫合": SEW}, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True), encoding="utf-8")
        print(f"[{g}] {json.dumps(report, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
