# -*- coding: utf-8 -*-
"""
同一個身材、不同款式：示範款式參數（design.yaml）能表現哪些上衣。

這裡的款式參數是人工指定的，不是 ChatGarment 看照片產生的；
用意是先看 GarmentCode 這套參數做得出哪些款式。每件都會生成網格並縫合。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\style_examples.py

輸出：output/chatgarment/style_<款式>/，每件約 1 分鐘（大部分花在縫合）
"""
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import design_to_mesh as dm  # noqa: E402  （匯入時會切到 GarmentCodeRC 根目錄）
import sew  # noqa: E402

BODY = dm.GC_ROOT / "assets" / "bodies" / "mean_male"

# 以 ChatGarment 範例襯衫為底，只改列出來的參數（數值都在 GarmentCode 允許的範圍內）
STYLES = {
    "long_sleeve_fitted": ("合身長袖上衣", {
        "meta": {"upper": "FittedShirt"},
        "shirt": {"length": 1.4},
        "sleeve": {"length": 1.0, "end_width": 0.7, "sleeve_angle": 20},
    }),
    "vneck_loose": ("寬鬆 V 領短袖", {
        "shirt": {"width": 1.25, "length": 1.4},
        "collar": {"f_collar": "VNeckHalf", "fc_depth": 1.3},
        "sleeve": {"length": 0.35, "end_width": 1.2},
    }),
    "tank_top": ("方領無袖背心", {
        "meta": {"upper": "FittedShirt"},
        "shirt": {"length": 1.5},
        "collar": {"f_collar": "SquareNeckHalf", "b_collar": "SquareNeckHalf", "fc_depth": 0.9, "width": 0.5},
        "sleeve": {"sleeveless": True},
    }),
    "oversize_long": ("長版寬鬆 T 恤", {
        "shirt": {"width": 1.3, "length": 2.2, "flare": 1.15},
        "sleeve": {"length": 0.5, "end_width": 1.4},
    }),
    "open_cardigan": ("開襟長袖外套", {
        "shirt": {"openfront": True, "width": 1.2, "length": 1.8},
        "collar": {"f_collar": "VNeckHalf", "fc_depth": 1.6},
        "sleeve": {"length": 1.05, "end_width": 0.9},
    }),
    "flare_blouse": ("傘狀下襬短袖", {
        "shirt": {"flare": 1.5, "length": 1.6},
        "collar": {"f_collar": "CurvyNeckHalf", "fc_depth": 0.6},
        "sleeve": {"length": 0.25, "end_width": 1.5, "sleeve_angle": 30},
    }),
}


def apply_overrides(base, overrides):
    design = copy.deepcopy(base)
    for section, params in overrides.items():
        for key, value in params.items():
            design[section][key]["v"] = value
    return design


def main():
    dm.ensure_system_json()
    base = yaml.safe_load(dm.DESIGN.read_text())["design"]
    only = set(sys.argv[1:])  # 可以只跑指定的款式

    for key, (title, overrides) in STYLES.items():
        if only and key not in only:
            continue
        name = f"style_{key}"
        print(f"\n[{title}] {name}")
        body, box, paths = dm.generate(apply_overrides(base, overrides), BODY.with_suffix(".yaml"), name)
        print(f"  衣服周長 {dm.pattern_girth(paths.in_g_spec):.1f} cm（身體胸圍 {body.params['bust']:.1f}）")
        sew.main(paths.g_box_mesh.parent, BODY.with_suffix(".obj"))


if __name__ == "__main__":
    main()
