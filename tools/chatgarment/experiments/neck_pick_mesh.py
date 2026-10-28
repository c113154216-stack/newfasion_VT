# -*- coding: utf-8 -*-
"""
第三批（領口比對）挑幾件生成網格：款式修正（to_design.py 已寫進 design_*.yaml）＋領深校正＋新縫合。
衣長維持 AI 的（這批只看領口，不做衣長校正），只取上衣。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\neck_pick_mesh.py [衣服資料夾名稱…]
不給名稱就做 PICKS。輸出：output/chatgarment/aibf16_i_neck_<衣服>/，網格檢視網站看得到
加 --flat-lapel：翻領改成平躺（lapel_standing＝False），輸出到 aibf16_i_lapel_<衣服>/
"""
import json
import sys
from pathlib import Path

import igl
import yaml

CWD = Path.cwd()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from hem_mesh import SEW, build  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402

RESULTS = CWD / "results" / "chatgarment_ai" / "bf16_i"
TAG = "aibf16_i_neck"
PICKS = [  # 每件代表一種情況（見 bf16_i/report_neckline.md）
    "i01_blue_crew_tee",               # 圓領：多出的高領被修正拿掉
    "i11_green_scoop_tee",             # 大圓領：AI 描述成圓領，領深被改成普通圓領
    "i18_green_vneck_tee",             # V 領：參數給圓領，修正成 V 領
    "i24_black_square_neck_long_sleeve_top",  # 方領：AI 描述成圓領，救不了
    "i29_green_boat_neck_sweater",     # 船領：GarmentCode 沒有船領，AI 給圓(曲)
    "i33_navy_mock_neck_tee",          # 半高領：修正規則把高領拿掉（改錯）
    "i40_charcoal_polo",               # Polo：修正補上翻領
    "i43_burgundy_dress_shirt",        # 襯衫：高領改成翻領
]


def main(*args):
    flat = "--flat-lapel" in args
    only = [a for a in args if not a.startswith("--")]
    tag = "aibf16_i_lapel" if flat else TAG
    dm.ensure_system_json()
    body_yaml = dm.GC_ROOT / "assets" / "bodies" / "mean_male.yaml"
    body = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_yaml.with_suffix(".obj")))
    body_mesh = (bv * 100, bf)
    for g in only or PICKS:
        src = next(RESULTS / g / f"design_{p}.yaml" for p in ("upper", "wholebody")
                   if (RESULTS / g / f"design_{p}.yaml").exists())
        design = yaml.safe_load(src.read_text(encoding="utf-8"))["design"]
        design["meta"]["bottom"]["v"] = None
        design["meta"]["wb"]["v"] = None
        if not design["meta"]["upper"]["v"]:
            design["meta"]["upper"]["v"] = "Shirt"
        if flat:  # Polo、襯衫的翻領是往下翻的；True 時後領是一條立著的布帶（跟高領一樣）
            design["collar"]["component"]["lapel_standing"]["v"] = False
        neck = fix_neck_depth(design, body, MetaGarment)
        folder, hem = build(design, f"{tag}_{g}", body_yaml, body_mesh)
        (folder / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True), encoding="utf-8")
        (folder / "hem_fix.json").write_text(json.dumps({"衣長校正": "這批只看領口，衣長不改", "領口校正": neck, "縫合": SEW},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{g}] 領口校正 {neck}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
