# -*- coding: utf-8 -*-
"""
從 T-pose 身體網格量出 GarmentCode 打版要的 26 個身體欄位（body_measurements.yaml）。

GarmentCodeRC 沒有附量測程式，每個欄位的精確量法也沒寫清楚，所以用「比例校正」：
  1. 寫一套只靠幾何的量法（水平截面、輪廓寬度），不依賴網格拓撲，SMPL 和 SMPL-X 都能量
  2. 拿 GarmentCode 附的 SMPL 女性範本（f_smpl_template，26 個欄位已知、也是 T-pose）當基準，
     用同一套量法量基準和目標身體
  3. 目標欄位 ＝ 基準欄位 × 目標量測 ／ 基準量測
這樣即使我們的量法和 GarmentCode 原本的定義不完全一樣，也會在比例中抵銷；
重要的是兩個身體用「同一套」量法。角度類欄位（肩斜、臀斜）沿用基準值，手臂角度設 0（T-pose）。

圍度用水平截面的凸包周長（＝皮尺繞一圈），和紀泓宇的量法一致：
胸圍＝腋下往下一段內的最大值、腰圍＝臀與胸之間的最小值、臀圍＝胯下往上第一個局部極大。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\measure_body.py [紀泓宇的量測 JSON]
不給 JSON 就量 SMPL-X 女性 betas 全 0。
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
GC_BODIES = REPO / "third_party" / "GarmentCodeRC" / "assets" / "bodies"
REFERENCE = GC_BODIES / "f_smpl_template"  # 基準：GarmentCode 附的 SMPL 女性範本（T-pose、公尺）
STEP = 0.5  # 掃描間距（cm）


def loops_at(mesh, y):
    """高度 y 的水平截面：每個封閉輪廓的 (x, z) 點（cm）。"""
    sec = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if sec is None:
        return []
    return [d[:, [0, 2]] for d in sec.discrete if len(d) >= 3]


def girth(pts):
    """凸包周長（皮尺繞一圈）。"""
    hull = ConvexHull(pts)
    p = pts[hull.vertices]
    return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())


def central(loops, max_offset=6.0):
    """中間那個輪廓（軀幹／脖子），沒有就回傳 None。"""
    near = [p for p in loops if abs(p[:, 0].mean()) < max_offset]
    return max(near, key=lambda p: np.ptp(p[:, 0])) if near else None


def measure(v_cm, faces):
    """幾何量測（cm）。v_cm：T-pose 身體，+y 上、手臂沿 ±x。"""
    mesh = trimesh.Trimesh(v_cm, faces, process=False)
    y0, y1 = v_cm[:, 1].min(), v_cm[:, 1].max()
    ys = np.arange(y0 + 1, y1 - 1, STEP)
    rows = []
    for y in ys:
        loops = loops_at(mesh, y)
        c = central(loops)
        rows.append({"y": y, "n": len(loops), "c": c,
                     "w": np.ptp(c[:, 0]) if c is not None else 0.0,
                     "g": girth(c) if c is not None else 0.0})

    # 胯下：從腳往上，第一個「中間有一整圈」的高度
    crotch_i = next(i for i, r in enumerate(rows) if r["c"] is not None and r["w"] > 15)
    # 腋下：軀幹往上，寬度 1 cm 內突然變寬超過 10%（截面開始接上手臂）之前的最後一個高度
    k = int(1 / STEP)
    armpit_i = next(i for i in range(crotch_i + 20 * k, len(rows))
                    if rows[i]["w"] > 1.10 * rows[i - k]["w"]) - k
    # 手臂頂：最後一個寬度還很寬（含手臂）的高度；之上只剩脖子和頭
    arm_top_i = max(i for i, r in enumerate(rows) if r["w"] > 60)
    # 脖子：手臂頂往上 15 cm 內周長最小處
    neck_rng = range(arm_top_i + 1, min(len(rows), arm_top_i + int(15 / STEP)))
    neck_i = min(neck_rng, key=lambda i: rows[i]["g"] if rows[i]["g"] > 0 else 1e9)
    # 胸圍：腋下往下 20 cm 內最大（SMPL-X 的軀幹周長一路往上變大，實際上就是緊貼腋下那一圈；
    # 紀泓宇的範例使用者這樣量是 86.7，他量 86.04）
    bust_i = max(range(armpit_i - int(20 / STEP), armpit_i + 1), key=lambda i: rows[i]["g"])
    # 臀圍：胯下往上第一個局部極大（周長先平滑）
    g = np.array([r["g"] for r in rows])
    gs = np.convolve(g, np.ones(5) / 5, mode="same")
    hip_i = next(i for i in range(crotch_i + 2, armpit_i) if gs[i] >= gs[i - 1] and gs[i] > gs[i + 1])
    # 腰圍：臀與腋下之間最小
    waist_i = min(range(hip_i + 1, armpit_i), key=lambda i: rows[i]["g"])
    # 胸點高度（bust_line 用）：腋下到腰的 35% 處。用比例而不是找最凸點，
    # 因為有些體型（例如紀泓宇的範例使用者）最凸的地方是肚子
    bust_pt_i = armpit_i - round(0.35 * (armpit_i - waist_i))

    # 大腿：胯下往下 3 cm 的單腳截面
    legs = loops_at(mesh, rows[crotch_i]["y"] - 3)
    thigh = max(girth(p) for p in legs) if legs else 0.0
    # 手臂長：左右指尖總寬扣掉腋下的軀幹寬，取一半
    span = np.ptp(v_cm[:, 0])
    torso_w = rows[armpit_i - 4]["w"]
    arm_len = (span - torso_w) / 2
    # 手腕：沿手臂（x 方向）切，每刀取最大的輪廓（手臂／手掌，不是手指），
    # 從手肘往指尖走，周長最小、而且還沒進到手指（> 8 cm）的地方
    xmax = v_cm[:, 0].max()
    wrist = 1e9
    for x in np.linspace(0.55 * xmax, 0.90 * xmax, 36):
        sec = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
        if sec is None:
            continue
        g_x = max((girth(d[:, [1, 2]]) for d in sec.discrete if len(d) >= 3), default=0.0)
        if g_x > 8:
            wrist = min(wrist, g_x)

    y = lambda i: rows[i]["y"]  # noqa: E731
    levels = {"胯下": y(crotch_i), "臀圍": y(hip_i), "腰圍": y(waist_i), "胸點": y(bust_pt_i),
              "腋下（胸圍）": y(armpit_i), "脖子": y(neck_i)}
    profile = (np.array([r["y"] for r in rows]), np.array([r["g"] for r in rows]))
    return {
        "_levels": levels, "_profile": profile,  # 畫圖檢查用，不參與換算
        "height": y1 - y0,
        "head": y1 - y(neck_i),
        "neck_w": rows[neck_i]["w"],
        "torso_w": torso_w,
        "neck_to_armpit": y(neck_i) - y(armpit_i),
        "neck_to_waist": y(neck_i) - y(waist_i),
        "neck_to_bust": y(neck_i) - y(bust_pt_i),
        "waist_to_hip": y(waist_i) - y(hip_i),
        "hip_to_crotch": y(hip_i) - y(crotch_i),
        "bust": rows[bust_i]["g"],
        "waist": rows[waist_i]["g"],
        "hips": rows[hip_i]["g"],
        "thigh": thigh,
        "arm": arm_len,
        "wrist": wrist,
    }


# GarmentCode 欄位 ← 用哪一個量測做比例（None＝沿用基準值）
FIELD_FROM = {
    "height": "height", "head_l": "head",
    "neck_w": "neck_w", "shoulder_w": "torso_w", "back_width": "torso_w",
    "armscye_depth": "neck_to_armpit", "shoulder_incl": None,
    "arm_length": "arm", "wrist": "wrist",
    "waist": "waist", "waist_back_width": "waist",
    "waist_line": "neck_to_waist", "waist_over_bust_line": "neck_to_waist",
    "bust_line": "neck_to_bust", "bust": "bust", "bust_points": "bust", "underbust": "bust",
    "hips": "hips", "hip_back_width": "hips", "bum_points": "hips",
    "hips_line": "waist_to_hip", "hip_inclination": None,
    "crotch_hip_diff": "hip_to_crotch", "leg_circ": "thigh",
}


def reference():
    import igl
    v, f = igl.read_triangle_mesh(str(REFERENCE.with_suffix(".obj")))
    body = yaml.safe_load(REFERENCE.with_suffix(".yaml").read_text())["body"]
    return measure(v * 100, f), body


def garmentcode_body(v_m, faces):
    """T-pose 身體（公尺）→ GarmentCode body 欄位 dict（cm／度），以及兩邊的原始量測。"""
    ref_m, ref_body = reference()
    tgt_m = measure(np.asarray(v_m) * 100, faces)
    body = {}
    for field, ref_val in ref_body.items():
        key = FIELD_FROM.get(field)
        body[field] = float(ref_val) if key is None else float(ref_val) * tgt_m[key] / ref_m[key]
    body["arm_pose_angle"] = 0  # T-pose
    return body, ref_m, tgt_m


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    import smplx_body
    if len(sys.argv) > 1:
        v, f, _ = smplx_body.from_measurement_json(sys.argv[1])
    else:
        v, f, _ = smplx_body.make_body("female")
    body, ref_m, tgt_m = garmentcode_body(v, f)
    print(f"{'量測':16s}{'基準':>8s}{'目標':>8s}")
    for k in ref_m:
        if not k.startswith("_"):
            print(f"{k:16s}{ref_m[k]:8.1f}{tgt_m[k]:8.1f}")
    print("\nGarmentCode 欄位：")
    print(yaml.safe_dump({"body": {k: round(float(val), 2) for k, val in body.items()}}, allow_unicode=True))


if __name__ == "__main__":
    main()
