# -*- coding: utf-8 -*-
r"""試驗：圓領 T 加一圈領圈（像 T 恤的羅紋領）。
GarmentCode 沒有專門的羅紋領，最接近的是 Turtle：沿領口縫一圈直條布（寬度 collar.component.depth，2～8 cm）。
  A：現在的交付版，沒有領圈（fix_collar 把圓領的領子配件清成無）
  B：加 Turtle、寬 2 cm，同時把領深多挖 2 cm（前、後都是），讓布條立起來後上緣回到原本領口的高度
兩個都走完整的 export_neutral（領深、衣長、袖長校正、縫合、攤平尖角、匯出）。

跑法（在 repo 根目錄；這台筆電要從英文路徑副本跑）：
    .venv\Scripts\python flows\body_fit\experiments\neckband_test.py A|B [AI 結果資料夾，預設 i10]
輸出：output/physics_export/neckband/<A|B>/<衣服>/
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

BAND_CM = 2.0
DEFAULT = en.dm.REPO / "results" / "chatgarment_ai" / "bf16_i" / "i10_red_crew_long_sleeve_tee"


def main(variant, src=None):
    src = Path(src) if src else DEFAULT
    out = en.dm.REPO / "output" / "physics_export" / "neckband" / variant
    # 兩個版本同時跑時，縫合的暫存資料夾名稱不能撞：複製一份改名的 AI 結果
    work = out / "_src" / f"{src.name}_{variant}"
    if not work.exists():
        shutil.copytree(src, work)
    en.OUT = out
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    body = BodyParameters(str(body_yaml))
    if variant == "B":
        # 領深最淺值加 2 cm（AI 給的深度都接近最淺，等於整個領口往下挖 2 cm）
        neckline.FRONT_MIN_RATIO += BAND_CM / body["neck_w"]
        neckline.BACK_MIN_RATIO += BAND_CM / body["neck_w"]
        load = en.load_design

        def load_with_band(result_dir):
            design = load(result_dir)
            design["collar"]["component"]["style"]["v"] = "Turtle"
            design["collar"]["component"]["depth"]["v"] = BAND_CM
            return design
        en.load_design = load_with_band
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    v0, f0, weights = smplx_body.make_body("neutral")
    en.export_one(work, body, body_yaml, (bv * 100, bf), (v0, f0, weights), lift, sleevelen.arms(lift))


if __name__ == "__main__":
    main(*sys.argv[1:])
