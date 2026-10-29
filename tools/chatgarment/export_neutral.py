# -*- coding: utf-8 -*-
"""
把一批 AI 結果全部照 SMPL-X 中性身體縫好，匯出給紀泓宇的 engine（取代 export_for_physics.py 的女性模板版）。

每件衣服：
  1. 款式：results/ 底下的 design_*.yaml（to_design.py 產生，已含領型、領口形狀、無袖、縮口修正），只取上衣
  2. 領深校正（neckline.py，照 mean_all 尺寸重算）
  3. 先用 AI 原本的衣長縫一次，量「縫合往上提」多少
  4. 衣長校正（hemline.py）：目標下襬 ＝（AI 縫好的下襬 ＋ 款式規則下襬）÷ 2，差超過 1 cm 再修正一次
  5. 縫合：新縫合（拉長超過 2% 變硬、彎曲剛度 1）＋壓短限制 2%（袖子不被擠短）
  6. 匯出：公尺、往下移回 SMPL-X 原始座標（和 engine 的身體同一個座標，不用再對齊）、
     固定點 neck_shoulder_band 照 engine/cloth/attachment.py 的規則（SMPL-X 蒙皮權重最大的關節是頸或左右鎖骨、離身體 1 cm 內）

身體：SMPL-X 中性、θ=0、β 全 0；打版尺寸檔用 GarmentCode 的 mean_all（手臂角度 0），見 export_for_physics.prepare_smplx_body。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\export_neutral.py results\\chatgarment_ai\\bf16_3090 [只做的資料夾…]
輸出：output/physics_export/neutral/<衣服>/<衣服>.obj、_pins.json、fix.json、design_used.yaml
"""
import ast
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import igl
import numpy as np
import yaml

CWD = Path.cwd()
sys.path.insert(0, str(Path(__file__).parent))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
import smplx_body  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from export_for_physics import pin_vertices, prepare_smplx_body  # noqa: E402
from hemline import correct, pattern_hem, plan  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions  # noqa: E402

OUT = dm.REPO / "output" / "physics_export" / "neutral"
SEW = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0, "compress_limit": 0.02}
RESOLUTION = 2.0  # 頂點間距（cm），約 2,000～4,000 頂點
TOLERANCE_CM = 1.0
YOKE_JOINTS = (12, 13, 14)  # 頸、左右鎖骨（和 engine/cloth/attachment.py 相同）
CONTACT_M = 0.01


def description(result_dir):
    text = (result_dir / "output.txt").read_text(encoding="utf-8")
    desc = ast.literal_eval(text.split("[描述]")[1].split("[版型]")[0].strip())
    garment = desc.get("upperbody_garment") or desc.get("wholebody_garment") or {}
    return garment.get("geometry_styles", {}), garment.get("garment_name")


def load_design(result_dir):
    """上衣在 design_upper.yaml；連身款（例如背心被判成 wholebody）在 design_wholebody.yaml。"""
    for part in ("upper", "wholebody"):
        src = result_dir / f"design_{part}.yaml"
        if src.exists():
            design = yaml.safe_load(src.read_text(encoding="utf-8"))["design"]
            design["meta"]["bottom"]["v"] = None
            design["meta"]["wb"]["v"] = None
            return design
    return None


def build(design, name, body_yaml, body_mesh):
    """生成版片、縫合；回傳 (資料夾, 縫好的頂點 cm, 面)。"""
    _, _, paths = dm.generate(design, body_yaml, name, resolution=RESOLUTION)
    v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
    edges = mesh_edges(f)
    rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
    x = sew(v, edges, rest, body=body_mesh, **SEW)
    write_obj_with_new_positions(paths.g_box_mesh, paths.g_box_mesh.with_name(f"{name}_sewn.obj"), x)
    return paths.g_box_mesh.parent, x, f


def band_pins(x, bv, bf, weights):
    """衣服頂點（公尺、SMPL-X 原始座標）中，最近身體面三個角的主導關節都是頸或鎖骨、且離身體 1 cm 內的。"""
    yoke = np.isin(weights.argmax(axis=1), YOKE_JOINTS)
    d2, face, _ = igl.point_mesh_squared_distance(x, bv, bf)
    return np.flatnonzero(yoke[bf[face]].all(axis=1) & (np.sqrt(d2) <= CONTACT_M)).tolist()


def export_one(result_dir, body, body_yaml, body_mesh, smplx, lift):
    g = result_dir.name
    design = load_design(result_dir)
    if design is None or not design["meta"]["upper"]["v"]:
        print(f"[{g}] 沒有上衣，略過", flush=True)
        return None
    t0 = time.time()
    styles, garment_name = description(result_dir)
    neck = fix_neck_depth(design, body, MetaGarment)

    # 先用 AI 的衣長縫一次，量縫合往上提多少。合身款不看衣長參數，先改成一般款再量
    probe = copy.deepcopy(design)
    fitted = probe["meta"]["upper"]["v"] == "FittedShirt"
    if fitted:
        probe["meta"]["upper"]["v"] = "Shirt"
    _, x, _ = build(probe, f"nx_{g}_base", body_yaml, body_mesh)
    base_hem = float(x[:, 1].min())
    rise = base_hem - pattern_hem(probe, body, MetaGarment)

    info = plan(design, body, MetaGarment, styles, garment_name,
                ai_sewn_hem=None if fitted else base_hem, lift=rise)
    name = f"nx_{g}"
    folder, x, f = build(design, name, body_yaml, body_mesh)
    hem = float(x[:, 1].min())
    if info is None:
        hem_report = "沒有對應的規則，衣長不改"
    else:
        passes = [round(hem, 1)]
        if abs(hem - info["_target"]) > TOLERANCE_CM:
            correct(design, info, hem)
            folder, x, f = build(design, name, body_yaml, body_mesh)
            hem = float(x[:, 1].min())
            passes.append(round(hem, 1))
        info["縫好的下襬 cm（每一輪）"] = passes
        info["最終衣長參數"] = round(float(design["shirt"]["length"]["v"]), 3)
        hem_report = {k: v for k, v in info.items() if not k.startswith("_")}

    # 匯出：公分 → 公尺，往下移回 SMPL-X 原始座標
    xm = x / 100.0 - [0.0, lift, 0.0]
    dst = OUT / g
    dst.mkdir(parents=True, exist_ok=True)
    write_obj_with_new_positions(folder / f"{name}_sewn.obj", dst / f"{g}.obj", xm)
    for extra in ("_material.mtl", "_texture_fabric.png"):
        src = folder / f"{name}{extra}"
        if src.exists():
            (dst / f"{g}{extra}").write_bytes(src.read_bytes())
    obj = (dst / f"{g}.obj").read_text()
    (dst / f"{g}.obj").write_text(obj.replace(f"{name}_material.mtl", f"{g}_material.mtl"))
    mtl = dst / f"{g}_material.mtl"
    if mtl.exists():
        mtl.write_text(mtl.read_text().replace(f"{name}_texture_fabric.png", f"{g}_texture_fabric.png"))

    bv, bf, weights = smplx
    band = band_pins(xm, bv, bf, weights)
    neckline, shoulder = pin_vertices(folder, name)
    dist = np.sqrt(igl.point_mesh_squared_distance(xm, bv, bf)[0]) * 100
    inside = igl.fast_winding_number_for_meshes(bv, bf, xm) > 0.5
    check = {"穿進身體的頂點%": round(100 * float(inside.mean()), 2),
             "離身體距離中位數 cm": round(float(np.median(dist)), 1),
             "下襬離地 cm": round(hem, 1)}
    (dst / f"{g}_pins.json").write_text(json.dumps({
        "units": "m", "up_axis": "+y", "front_axis": "+z",
        "coordinates": "SMPL-X 原始座標（neutral、θ=0、β 全 0，和 engine 的身體相同，不用再平移對齊）",
        "pose": "T-pose（SMPL-X θ=0）",
        "n_vertices": int(len(xm)), "n_faces": int(len(f)), "has_uv": True, "resolution_cm": RESOLUTION,
        "pins": {"neck_shoulder_band": band,  # 建議用這組：和 engine/cloth/attachment.py 的 yoke_vertices 同規則
                 "neckline": neckline, "shoulder": shoulder,
                 "neckline_and_shoulder": sorted(set(neckline) | set(shoulder))},
        "check": check,
        "note": "頂點編號 0 起算，順序和 OBJ 的 v 行相同",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (dst / "fix.json").write_text(json.dumps(
        {"衣長校正": hem_report, "領口校正": neck, "縫合": SEW, "縫合往上提 cm": round(rise, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (dst / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True), encoding="utf-8")
    print(f"[{g}] {len(xm)} 頂點、固定點 {len(band)}、下襬 {hem:.1f} cm、穿進身體 {check['穿進身體的頂點%']}%"
          f"（{time.time() - t0:.0f} 秒）", flush=True)
    return check


def main(results, *only):
    dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    body = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body_mesh = (bv * 100, bf)
    v0, f0, weights = smplx_body.make_body("neutral")
    for result_dir in sorted(p for p in (CWD / results).iterdir() if p.is_dir()):
        if only and result_dir.name not in only:
            continue
        try:
            export_one(result_dir, body, body_yaml, body_mesh, (v0, f0, weights), lift)
        except Exception as e:  # 一件失敗不中斷整批
            print(f"[{result_dir.name}] 失敗：{e!r}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
