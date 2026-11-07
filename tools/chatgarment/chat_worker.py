# -*- coding: utf-8 -*-
"""
chat_server.py 的幫手，在 GarmentCode 的 .venv 執行（ChatGarment 的環境沒有 GarmentCode 的套件）。

    python chat_worker.py simplify <design.yaml>             舊款式簡化成文字（修改衣服時給模型看）
    python chat_worker.py build <問答資料夾> [--sew] [--raw] [--lower]
        模型輸出 → design_*.yaml → 2D 版片＋3D 網格（--sew 再縫到 mean_male 身上）
        --raw：不套用我們的修正（領子、領口形狀、無袖、縮口、領深），看 ChatGarment 原本的樣子
        --lower：下身也做（計畫書範圍只有上衣，預設只做上衣）
最後一行印出 JSON，chat_server.py 只讀這一行。
"""
import argparse
import contextlib
import copy
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}  # 和 neck_general_mesh.py 相同的新縫合


def flat(design, prefix=""):
    """款式參數攤平成 {"sleeve.length": 0.78, …}，網頁用表格顯示。"""
    out = {}
    for k, v in design.items():
        if isinstance(v, dict) and "v" in v:
            out[prefix + k] = v["v"]
        elif isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
    return out


def simplify(path):
    import to_design
    gu = to_design.load_chatgarment_utils()
    return {"text": gu.get_simplified_json_config(str(Path(path).resolve()))}


def build(folder, do_sew, raw, lower):
    import to_design
    folder = Path(folder).resolve()
    if raw:  # 不修正：把修正函式換成什麼都不做
        for name in ("fix_collar", "fix_neck_shape", "fix_sleeveless", "fix_cuff"):
            setattr(to_design, name, lambda design, styles: None)
    gu = to_design.load_chatgarment_utils()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        designs = to_design.to_designs(gu, folder)
    fixes = [line.split("] ", 1)[-1] for line in log.getvalue().splitlines() if line.strip()]
    for part, design in designs.items():
        (folder / f"design_{part}.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True))

    os.environ["GC_OUTPUT"] = str(folder)  # design_to_mesh 匯入時讀這個決定輸出位置
    import design_to_mesh as dm  # 匯入時切到 GarmentCodeRC 根目錄
    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs.meta_garment import MetaGarment
    from neckline import fix_neck_depth
    dm.ensure_system_json()
    body_yaml = dm.GC_ROOT / "assets" / "bodies" / "mean_male.yaml"
    body = BodyParameters(str(body_yaml))

    parts = []
    for part, design in designs.items():
        if part == "lower" and not lower:
            continue
        t = time.time()
        design = copy.deepcopy(design)
        if part == "upper":  # 只做上衣：下身與腰帶關掉（和 to_design.py 相同）
            design["meta"]["bottom"]["v"] = None
            design["meta"]["wb"]["v"] = None
        if part in ("upper", "wholebody") and design["meta"]["upper"]["v"] and not raw:
            with contextlib.redirect_stdout(io.StringIO()):
                neck = fix_neck_depth(design, body, MetaGarment)
            if neck:
                fixes.append(f"領深 {neck['參數（前, 後）']}（目標 {neck['目標深度 cm']} cm）")
        name = f"{folder.name}_{part}"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, paths = dm.generate(design, body_yaml, name)
        except Exception as e:  # 款式組不出版片（參數互相矛盾）也回報，不中斷其他部位
            parts.append({"part": part, "error": f"{type(e).__name__}: {e}", "params": flat(design)})
            continue
        mesh = paths.g_box_mesh
        sewn = False
        if do_sew:
            import igl
            from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions
            bv, bf = igl.read_triangle_mesh(str(body_yaml.with_suffix(".obj")))
            v, f = igl.read_triangle_mesh(str(mesh))
            edges = mesh_edges(f)
            rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
            x = sew(v, edges, rest, body=(bv * 100, bf), **SEW)
            out = mesh.with_name(mesh.name.replace("_boxmesh.obj", "_sewn.obj"))
            write_obj_with_new_positions(mesh, out, x)
            mesh, sewn = out, True
        rel = lambda p: f"{folder.name}/{p.relative_to(folder).as_posix()}"
        png = next(mesh.parent.glob("*_pattern.png"), None)
        parts.append({"part": part, "summary": to_design.summary(design), "params": flat(design),
                      "pattern": rel(png) if png else None, "obj": rel(mesh), "sewn": sewn,
                      "seconds": round(time.time() - t, 1)})
    return {"parts": parts, "fixes": fixes, "raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["simplify", "build"])
    ap.add_argument("path")
    ap.add_argument("--sew", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--lower", action="store_true")
    args = ap.parse_args()
    out = simplify(args.path) if args.cmd == "simplify" else build(args.path, args.sew, args.raw, args.lower)
    print(json.dumps(out, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
