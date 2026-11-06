# -*- coding: utf-8 -*-
"""
袖長規則：AI 描述是長袖（geometry_styles.sleeves 是 "long sleeves"）時，把 sleeve.length 調到縫好後袖口剛好到手腕；
短袖、無袖照 AI 的參數，不改。

為什麼：32 件長袖縫好後袖口離手腕平均 −7.1 cm，和 AI 的袖長參數（0.78～0.97）幾乎成正比
（袖口離手腕 ≈ 49.2 × 參數 − 50.4，相關係數 0.87；參數 1.0 時差 −1.3 cm），縫合本身沒有把袖子縮短。
之前以為「縫合把袖子往肩膀拉短 5～7 cm」，是拿縫合前的 box mesh 當基準：GarmentCode 縫合前把腋下縫線畫得比原長長約 9 cm、
袖口斜斜的，那不是袖子真正的位置（見 docs/筆記_袖子偏短與縮口.md 九）。
AI 參數偏小的可能原因：ChatGarment 學的 GarmentCodeData 是 A-pose＋重力，袖子會往手腕滑；我們是 T-pose、沒有重力。

量法：袖子（含袖口布）頂點沿手臂方向（SMPL-X 肩關節 → 手腕）最遠的位置，減掉手腕位置，左右平均（cm）。
先用 AI 的參數縫一次量袖口，用斜率 0.9 × arm_length（cm／參數，實測 49.2 ≈ 0.91 × 54）算新參數；
差超過 1 cm 再用兩次實測的斜率修正一次。
"""
import numpy as np

import smplx_body

TOLERANCE_CM = 1.0


def arms(lift):
    """{x 正負號: (肩關節, 手臂方向, 手腕在方向上的位置)}，單位 cm、腳底在 0（lift 是腳底往上移的公尺數）。"""
    model = smplx_body.load_model("neutral")
    v, _, _ = smplx_body.make_body("neutral")
    J = (model["J_regressor"] @ v + [0.0, lift, 0.0]) * 100
    out = {}
    for shoulder, wrist in ((16, 20), (17, 21)):
        c = J[shoulder]
        u = (J[wrist] - c) / np.linalg.norm(J[wrist] - c)
        out[1 if J[wrist, 0] > 0 else -1] = (c, u, float((J[wrist] - c) @ u))
    return out


def sleeve_mask(seg_lines):
    return np.array(["sleeve" in s or "cuff" in s for s in seg_lines])


def reach(x, sleeve, arm):
    """袖口離手腕（cm，正＝超過手腕）。沒有袖子時回傳 None。"""
    if sleeve.sum() < 10:
        return None
    vals = []
    for sign, (c, u, wrist) in arm.items():
        side = sleeve & (sign * x[:, 0] > 0)
        if side.sum() < 5:
            return None
        vals.append(((x[side] - c) @ u).max() - wrist)
    return float(np.mean(vals))


LONG = "long sleeves"
SLOPE = 0.9  # 袖長參數每加 1，袖口約往外 0.9 × arm_length cm
TARGET_CM = 0.0  # 長袖：袖口到手腕


def is_long(styles):
    sleeves = (styles or {}).get("sleeves") or []
    return bool(sleeves) and sleeves[0] == LONG


def plan(design, styles, after, arm_length):
    """用 AI 參數縫一次後呼叫：after＝縫好的袖口離手腕。長袖就地改 sleeve.length，回傳說明 dict；不適用回傳 None。"""
    if after is None or design["sleeve"]["sleeveless"]["v"] or not is_long(styles):
        return None
    L0 = float(design["sleeve"]["length"]["v"])
    L1 = L0 + (TARGET_CM - after) / (SLOPE * arm_length)
    design["sleeve"]["length"]["v"] = L1
    return {"規則": "長袖：袖口到手腕", "目標袖口離手腕 cm": TARGET_CM, "AI 參數縫好 cm": round(after, 1),
            "袖長參數": [round(L0, 3), round(L1, 3)], "_L0": L0, "_after0": after, "_target": TARGET_CM, "_arm": arm_length}


def correct(design, info, sewn):
    """改過參數縫好的袖口還差超過容許值時，用兩次實測的斜率再修一次（就地修改）。"""
    L1 = float(design["sleeve"]["length"]["v"])
    slope = (sewn - info["_after0"]) / (L1 - info["_L0"]) if abs(L1 - info["_L0"]) > 1e-6 else None
    # 實測斜率太小會外插過頭，至少取 0.3 × arm_length
    slope = max(slope or 0.0, 0.3 * info["_arm"])
    L2 = L1 + (info["_target"] - sewn) / slope
    design["sleeve"]["length"]["v"] = float(L2)
    info["袖長參數"].append(round(L2, 3))
    return L2
