# -*- coding: utf-8 -*-
r"""試驗：短袖圓領 T 有沒有加 Turtle 領圈的對比。兩件都先把領口加寬 5 cm、前領深 9.5 cm（見 neck_width_test.py，
現在的領口縫好後會爬到脖子上、太小），只差在領圈：
  plain：沒有領圈
  band ：加 Turtle（寬 2 cm），前後領深再多挖 2 cm，讓布條立起來後上緣回到 plain 的領口高度
都走完整的 export_neutral。

跑法（在 repo 根目錄；這台筆電要從英文路徑副本跑）：
    .venv\Scripts\python flows\body_fit\experiments\neckband_tee.py plain|band [AI 結果資料夾，預設 i01 短袖圓領 T]
輸出：output/physics_export/neckband_tee/<plain|band>/<衣服>/
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import export_neutral as en  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
import neckline  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
import igl  # noqa: E402
import smplx_body  # noqa: E402
import sleevelen  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402

WIDEN_CM = 5.0
FRONT_DEPTH_CM = 9.5
BAND_CM = 2.0
DEFAULT = en.dm.REPO / "results" / "chatgarment_ai" / "bf16_i" / "i01_blue_crew_tee"


def main(variant, src=None):
    assert variant in ("plain", "band")
    src = Path(src) if src else DEFAULT
    out = en.dm.REPO / "output" / "physics_export" / "neckband_tee" / variant
    work = out / "_src" / f"{src.name}_{variant}"
    if not work.exists():
        shutil.copytree(src, work)
    en.OUT = out
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    body = BodyParameters(str(body_yaml))
    lo, hi = body["neck_w"], body["_base_sleeve_balance"] - 2
    extra = BAND_CM if variant == "band" else 0.0
    neckline.FRONT_MIN_RATIO = (FRONT_DEPTH_CM + extra) / body["neck_w"]
    neckline.BACK_MIN_RATIO += extra / body["neck_w"]
    load = en.load_design

    def load_variant(result_dir):
        design = load(result_dir)
        p = design["collar"]["width"]["v"]
        design["collar"]["width"]["v"] = (lo + p * (hi - lo) + WIDEN_CM - lo) / (hi - lo)
        design["collar"]["component"]["style"]["v"] = "Turtle" if variant == "band" else None
        design["collar"]["component"]["depth"]["v"] = BAND_CM
        return design
    en.load_design = load_variant
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    v0, f0, weights = smplx_body.make_body("neutral")
    en.export_one(work, body, body_yaml, (bv * 100, bf), (v0, f0, weights), lift, sleevelen.arms(lift))


if __name__ == "__main__":
    main(*sys.argv[1:])
