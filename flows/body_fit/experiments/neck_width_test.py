# -*- coding: utf-8 -*-
r"""試驗：領口加寬，看縫好後領口會不會落回脖子根部。
現在的圓領 T（i10）版片上領寬（肩頸點到肩頸點）約 22 cm，但縫好後整圈往上爬到脖子較細的地方，
後領比脖子根部高 3 cm 多，開口被擠成寬 14 cm 的窄圈，看起來太小、勒脖子。
  W3：領寬加 3 cm、前領深 9.5 cm
  W5：領寬加 5 cm、前領深 9.5 cm
領寬用 GarmentCode 的 collar.width（0～1 在 neck_w 和 _base_sleeve_balance−2 之間線性內插）反推。
兩個都走完整的 export_neutral，和 neckband_test.py 的 A 比。

跑法（在 repo 根目錄；這台筆電要從英文路徑副本跑）：
    .venv\Scripts\python flows\body_fit\experiments\neck_width_test.py W3|W5 [AI 結果資料夾，預設 i10]
輸出：output/physics_export/neckwidth/<W3|W5>/<衣服>/
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

VARIANTS = {"W3": 3.0, "W5": 5.0}
FRONT_DEPTH_CM = 9.5
DEFAULT = en.dm.REPO / "results" / "chatgarment_ai" / "bf16_i" / "i10_red_crew_long_sleeve_tee"


def main(variant, src=None):
    src = Path(src) if src else DEFAULT
    out = en.dm.REPO / "output" / "physics_export" / "neckwidth" / variant
    work = out / "_src" / f"{src.name}_{variant}"
    if not work.exists():
        shutil.copytree(src, work)
    en.OUT = out
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    body = BodyParameters(str(body_yaml))
    lo, hi = body["neck_w"], body["_base_sleeve_balance"] - 2
    neckline.FRONT_MIN_RATIO = FRONT_DEPTH_CM / body["neck_w"]
    load = en.load_design

    def load_wider(result_dir):
        design = load(result_dir)
        p = design["collar"]["width"]["v"]
        width = lo + p * (hi - lo) + VARIANTS[variant]
        design["collar"]["width"]["v"] = (width - lo) / (hi - lo)
        print(f"領寬 {lo + p * (hi - lo):.1f} → {width:.1f} cm（參數 {p:.3f} → {design['collar']['width']['v']:.3f}）", flush=True)
        return design
    en.load_design = load_wider
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    v0, f0, weights = smplx_body.make_body("neutral")
    en.export_one(work, body, body_yaml, (bv * 100, bf), (v0, f0, weights), lift, sleevelen.arms(lift))


if __name__ == "__main__":
    main(*sys.argv[1:])
