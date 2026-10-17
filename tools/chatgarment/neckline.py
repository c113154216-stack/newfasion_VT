# -*- coding: utf-8 -*-
"""
領口深度校正：適用任何身體、任何上衣款式。

問題：GarmentCode 的前領深 ＝ fc_depth × 胸線高 ＋ 肩斜補正，而設計範圍規定 fc_depth ≥ 0.3。
     就算 AI 給最淺（0～1 的數值接近 0），前領也有 13 cm 以上，後領約 8 cm；
     一般圓領 T 恤前深約 8 cm、後深約 2 cm，領口一圈 48～54 cm。

做法：
1. 最淺的深度用身體頸寬的比例定（前 0.39、後 0.10 × neck_w；男性平均身材約 8 cm、2 cm），
   最深的沿用 GarmentCode 原本參數最大時的深度。
2. AI 的 0～1 數值原本代表「在 GarmentCode 範圍內多深」，改成線性對應到上面這個新範圍。
   所以 AI 判斷為淺圓領的會變成真正的圓領，判斷為深 V 領的照樣很深。
3. 不自己重寫 GarmentCode 的公式（款式不同，軀幹版片寬度不同，肩斜補正就不同）：
   直接用 GarmentCode 組出 2D 版片、量頸側點到前後中心的實際深度，量出「參數 → 深度」的線性關係再反推。
   只組版片、不生成網格，很快。

深度的定義（版片座標）：頸側點（前或後片最高的頂點）到中心線（x＝0）上領口點的高度差。
"""
import copy
import io
import contextlib

import numpy as np

# 最淺的領深 ＝ 比例 × 身體頸寬（neck_w）。8 cm／2 cm 是一般男 T 的打版尺寸，用平均男性的頸寬 20.3 cm 換算
FRONT_MIN_RATIO = 0.39
BACK_MIN_RATIO = 0.10
# GarmentCode 設計範圍（assets/design_params/default.yaml）
FC_RANGE = (0.3, 2.0)
BC_RANGE = (0.0, 2.0)


def _depths(design, body, MetaGarment):
    """組出版片，回傳（前深, 後深）cm；量不到（例如前開襟沒有中心線）回傳 None。"""
    with contextlib.redirect_stdout(io.StringIO()):
        pattern = MetaGarment("neck_probe", body, copy.deepcopy(design)).assembly()
    panels = pattern.pattern["panels"]

    def depth(suffix):
        names = [n for n in panels if n.endswith(suffix)]
        if not names:
            return None
        v = np.array(panels[names[0]]["vertices"], dtype=float)
        center = v[np.abs(v[:, 0]) < 1e-6]
        if len(center) == 0:
            return None
        return float(v[:, 1].max() - center[:, 1].max())

    f, b = depth("ftorso"), depth("btorso")
    return (f, b) if f is not None and b is not None else None


def fix_neck_depth(design, body, MetaGarment):
    """就地修改 design 的 collar.fc_depth／bc_depth，回傳說明 dict；不適用時回傳 None。
    body：GarmentCode 的 BodyParameters；MetaGarment：assets.garment_programs.meta_garment.MetaGarment"""
    if not design["meta"]["upper"]["v"] or design["shirt"].get("strapless", {}).get("v"):
        return None
    collar = design["collar"]
    p_f, p_b = float(collar["fc_depth"]["v"]), float(collar["bc_depth"]["v"])
    # AI 原本的相對深度（0＝最淺、1＝最深）
    u_f = float(np.clip((p_f - FC_RANGE[0]) / (FC_RANGE[1] - FC_RANGE[0]), 0, 1))
    u_b = float(np.clip((p_b - BC_RANGE[0]) / (BC_RANGE[1] - BC_RANGE[0]), 0, 1))

    def probe(pf, pb):
        d = copy.deepcopy(design)
        d["collar"]["fc_depth"]["v"], d["collar"]["bc_depth"]["v"] = pf, pb
        return _depths(d, body, MetaGarment)

    at0, at1, at_max = probe(0.0, 0.0), probe(0.5, 0.5), probe(FC_RANGE[1], BC_RANGE[1])
    if not (at0 and at1 and at_max):
        return None
    neck_w = float(body["neck_w"])
    targets, params = [], []
    for k, (u, lo_ratio) in enumerate([(u_f, FRONT_MIN_RATIO), (u_b, BACK_MIN_RATIO)]):
        d_min = lo_ratio * neck_w
        d_max = max(at_max[k], d_min)
        target = d_min + u * (d_max - d_min)
        slope = (at1[k] - at0[k]) / 0.5  # 深度 ＝ 參數 × 胸線高 ＋ 肩斜補正，在不碰上限時是線性的
        targets.append(target)
        params.append((target - at0[k]) / slope)

    # 再組一次確認；碰到上限而不線性時，用量到的誤差修正一次
    got = probe(*params)
    if got:
        for k in range(2):
            slope = (at1[k] - at0[k]) / 0.5
            params[k] += (targets[k] - got[k]) / slope
        got = probe(*params)

    collar["fc_depth"]["v"], collar["bc_depth"]["v"] = float(params[0]), float(params[1])
    return {
        "AI 相對深度（前, 後）": (round(u_f, 3), round(u_b, 3)),
        "目標深度 cm": tuple(round(t, 1) for t in targets),
        "實際深度 cm": tuple(round(g, 1) for g in got) if got else None,
        "參數（前, 後）": (round(p_f, 3), round(p_b, 3), "→", round(params[0], 3), round(params[1], 3)),
    }
