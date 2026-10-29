# -*- coding: utf-8 -*-
"""
T-pose（交給紀泓宇的那條路）下，袖子會不會也縫短？

實驗一、二 A 都在 A-pose（GarmentCode mean_male）做；交給物理的網格是 T-pose（export_for_physics.py，
SMPL 女性模板、arm_pose_angle＝0）。T-pose 手臂水平，重力跟手臂垂直，結論可能不同。
同實驗一的 3 件款式（design_used.yaml，已套用款式、衣長修正；領深照 T-pose 身體重算），在 T-pose 身體上生成，
舊縫合、新縫合各縫一次，量：袖口離手腕、領口拉長 %、下襬離地、穿進身體 %。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_tpose_test.py
輸出：output/sleeve_tpose/<縫法代號>_<衣服>_sewn.obj、results.json
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
from export_for_physics import prepare_tpose_body  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402
from sleeve_sew_ablation import GARMENTS, measure  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "output" / "sleeve_tpose"
METHODS = {
    "舊": ("old", {}),
    "新（現在）": ("new", {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}),
}
ARM_PARTS = ("Arm", "ForeArm", "Hand", "HandIndex1")  # SMPL 分區：上臂、前臂、手掌、手指


def arm_smpl(bv, side):
    """SMPL 拓撲（6,890 頂點）的手臂軸與手腕：同 sleeve_sew_ablation.arm，手臂頂點改用 SMPL 分區。"""
    seg = json.loads((dm.GC_ROOT / "assets" / "bodies" / "smpl_vert_segmentation.json").read_text())
    a = bv[sorted({i for p in ARM_PARTS for i in seg[f"{side}{p}"]})]
    c = a.mean(0)
    u = np.linalg.svd(a - c)[2][0]
    u = u if (u @ (a.mean(0) - bv.mean(0))) > 0 else -u  # 朝手指方向
    t = (a - c) @ u
    r = np.linalg.norm((a - c) - np.outer(t, u), axis=1)
    edges = np.linspace(t.min(), t.max(), 60)
    mid, rad = (edges[:-1] + edges[1:]) / 2, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        rad.append(np.percentile(r[m], 90) if m.sum() > 5 else np.inf)
    span = t.max() - t.min()
    ok = (mid > t.min() + 0.55 * span) & (mid < t.max() - 0.12 * span)
    return c, u, mid[ok][np.argmin(np.array(rad)[ok])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dm.ensure_system_json()
    body_obj, body_yaml = prepare_tpose_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    # measure() 用 x 正負分左右：key 的正負號＝那隻手在 x 的哪一邊
    arms = {}
    for side in ("left", "right"):
        c, u, w = arm_smpl(bv, side)
        arms[1 if c[0] > 0 else -1] = (c, u, w)
    results = {}
    for g in GARMENTS:
        short = g.split("_v2_")[1]
        design = yaml.safe_load((dm.OUT_ROOT / g / "design_used.yaml").read_text(encoding="utf-8"))["design"]
        # design_used.yaml 的領深參數是照 mean_male 倒推的，換身體要重算（不然版片可能組不起來）
        fix_neck_depth(design, BodyParameters(str(body_yaml)), MetaGarment)
        name = f"tpose_{short}"
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
        results[short] = {"縫合前": measure(v, sleeve, collar_e, rest, E, arms)}
        print(f"\n{short}　縫合前 {results[short]['縫合前']}", flush=True)
        for label, (code, kw) in METHODS.items():
            t0 = time.time()
            x = sew(v, E, rest, body=(bv, bf), **kw)
            igl.write_triangle_mesh(str(OUT / f"{code}_{short}_sewn.obj"), x, f)
            m = measure(x, sleeve, collar_e, rest, E, arms)
            m["穿進身體%"] = round(100 * float(np.mean(igl.signed_distance(x, bv, bf)[0] < 0)), 2)
            results[short][label] = m
            print(f"  {label:6s} {m}（{time.time() - t0:.0f} 秒）", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
