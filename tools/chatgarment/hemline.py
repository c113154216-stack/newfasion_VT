# -*- coding: utf-8 -*-
"""
衣長校正：依款式決定下襬位置，跟 AI 的判斷各取一半（2026-10-19 決定的「解法四」，見 docs/筆記_衣長偏短問題與解法.md）。

問題：ChatGarment 的衣長數字幾乎都落在 1.2 附近（訓練資料 70% 的上衣用預設值 1.2），
     下襬停在腰下一點；短版、長版 T 也都給 1.2 左右。但第一問的文字描述在「種類」上是對的。

做法：
1. 從第一問的描述判斷種類（T 恤、帽 T、毛衣）和長度詞（normal…）
2. 規則位置：臀圍線往下，「臀圍線到胯下」距離的 RULE_RATIO 倍（T 恤 0.5、帽 T／毛衣 0.3）
3. 目標下襬 ＝ (AI 的下襬 ＋ 規則位置) ÷ 2
   AI 判成合身款（FittedShirt）時，衣長參數根本沒被用到（合身款永遠只到腰），
   沒有可以保留的判斷 → 改成一般款（Shirt），目標直接用規則位置（合身款處理方式①）
4. 下襬是「縫好之後」的高度：新縫合會讓衣服掛到肩上、整件往上提，所以要扣掉往上提的量。
   版片下襬和衣長參數是直線關係，組兩次版片量出斜率再反推（不重寫 GarmentCode 的公式）。

規則表只訂了這批衣服用到的「一般長度」；其他種類、長度詞（短版、長版、襯衫、外套）回傳 None，不改。
"""
import contextlib
import copy
import io

import numpy as np

# 下襬規則：臀圍線往下，「臀圍線到胯下」距離的幾倍（0＝臀圍線，1＝胯下）
RULE_RATIO = {"tee": 0.5, "hoodie": 0.3, "sweater": 0.3}
CATEGORY_WORDS = [  # 依序比對，先比較具體的
    ("hoodie", ("hoodie", "sweatshirt")),
    ("sweater", ("sweater", "jumper", "pullover", "long sleeve", "long-sleeve")),
    ("tee", ("t-shirt", "tshirt", "t shirt", "tee", "polo")),
]
NORMAL_WORDS = ("normal", "regular", "medium", "standard", "hip")


def classify(styles, garment_name):
    """(種類, 長度詞)；判斷不出來或沒有規則時種類是 None。"""
    name = (garment_name or "").lower()
    category = next((c for c, words in CATEGORY_WORDS if any(w in name for w in words)), None)
    length = " ".join(styles.get("length") or []).lower() or "normal"
    return category, length


def body_levels(body):
    """尺寸表推算的高度（cm，腳底 0）：上衣頂端、腰、臀圍線、胯下。"""
    top = body["height"] - body["head_l"]
    waist = top - body["waist_line"]
    hip = waist - body["hips_line"]
    return {"頂端": top, "腰": waist, "臀圍線": hip, "胯下": hip - body["crotch_hip_diff"]}


def rule_hem(body, category):
    lv = body_levels(body)
    return lv["臀圍線"] - RULE_RATIO[category] * (lv["臀圍線"] - lv["胯下"])


def pattern_hem(design, body, MetaGarment):
    """組出版片，回傳軀幹版片下襬的高度（cm）。軀幹版片沒有旋轉，下襬 ＝ 位移 y ＋ 最低頂點 y。"""
    with contextlib.redirect_stdout(io.StringIO()):
        pattern = MetaGarment("hem_probe", body, copy.deepcopy(design)).assembly()
    ys = [p["translation"][1] + np.array(p["vertices"], float)[:, 1].min()
          for n, p in pattern.pattern["panels"].items() if n.endswith(("ftorso", "btorso"))]
    return float(min(ys))


def hem_slope(design, body, MetaGarment):
    """版片下襬隨衣長參數的變化（cm／參數），在 AI 的值附近量。"""
    L = float(design["shirt"]["length"]["v"])
    d = copy.deepcopy(design)
    d["shirt"]["length"]["v"] = L + 0.5
    return (pattern_hem(d, body, MetaGarment) - pattern_hem(design, body, MetaGarment)) / 0.5


def plan(design, body, MetaGarment, styles, garment_name, ai_sewn_hem=None, lift=None):
    """決定目標下襬並就地修改 design（衣長參數、合身款改一般款）。回傳說明 dict；不適用時回傳 None。
    ai_sewn_hem：用 AI 原本的衣長縫好之後的下襬（cm），一般款需要它來取平均。
    lift：縫合讓衣服往上提的量（cm）＝ 縫好的下襬 − 版片下襬；不知道時用 ai_sewn_hem 算。"""
    if not design["meta"]["upper"]["v"]:
        return None
    category, length = classify(styles, garment_name)
    if category is None or not any(w in length for w in NORMAL_WORDS):
        return None

    info = {"種類": category, "長度詞": length}
    y_rule = rule_hem(body, category)
    fitted = design["meta"]["upper"]["v"] == "FittedShirt"
    if fitted:
        design["meta"]["upper"]["v"] = "Shirt"  # 合身款不看衣長參數，改成一般款
        info["合身款"] = "FittedShirt → Shirt"

    L_ai = float(design["shirt"]["length"]["v"])
    y_ai_pattern = pattern_hem(design, body, MetaGarment)
    if lift is None:
        if ai_sewn_hem is None:
            raise ValueError("要給 lift 或 ai_sewn_hem")
        lift = ai_sewn_hem - y_ai_pattern

    if fitted or ai_sewn_hem is None:
        y_target = y_rule  # 合身款處理方式①：沒有可以保留的 AI 判斷
    else:
        y_target = (ai_sewn_hem + y_rule) / 2

    slope = hem_slope(design, body, MetaGarment)
    if abs(slope) < 1e-6:
        return None
    L_new = L_ai + (y_target - lift - y_ai_pattern) / slope
    design["shirt"]["length"]["v"] = float(L_new)
    info.update({
        "規則下襬 cm": round(y_rule, 1),
        "AI 縫好的下襬 cm": None if ai_sewn_hem is None else round(ai_sewn_hem, 1),
        "目標下襬 cm": round(y_target, 1),
        "縫合往上提 cm": round(lift, 1),
        "衣長參數": (round(L_ai, 3), "→", round(L_new, 3)),
        "_slope": slope, "_target": y_target,
    })
    return info


def correct(design, info, sewn_hem):
    """縫好之後量到的下襬跟目標差太多時，用斜率修正一次衣長參數（就地修改），回傳新的參數。"""
    L = float(design["shirt"]["length"]["v"]) + (info["_target"] - sewn_hem) / info["_slope"]
    design["shirt"]["length"]["v"] = float(L)
    return L
