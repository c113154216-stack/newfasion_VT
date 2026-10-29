# -*- coding: utf-8 -*-
"""
袖口縮口試驗：AI 描述寫 ribbed cuffs，版型卻是 cuff＝無。把縮口（CuffBand）加回去再縫，跟原本比較。

沿用 hem_mesh.py 做好的款式（已套用款式修正、衣長、領深校正，design_used.yaml），只改袖口，縫法相同（新縫合）。
GarmentCode 加袖口時會把袖子本體扣掉同樣長度（cuff_len × 手臂長），總袖長不變。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\cuff_test.py [來源資料夾名稱]
預設 aibf16_3090_v2_g03_blue_hoodie、top_ruffle 1（AI 給的值）。第二個參數可給 top_ruffle：
袖子末端收皺幾倍，縮口一圈＝袖子末端一圈 ÷ top_ruffle（最少手腕一圈），1 就是完全不收。
輸出：output/chatgarment/aibf16_cuff_<衣服>/（top_ruffle 不是 1 時資料夾名稱加 _r<值>）
"""
import json
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
from hem_mesh import SEW, build  # noqa: E402
from sleeve_sew_ablation import arm  # noqa: E402


def cuff_end(folder, name, arms):
    """袖口（或有縮口時縮口）最遠的點離手腕幾 cm，左右平均。"""
    v, _ = igl.read_triangle_mesh(str(folder / f"{name}_sewn.obj"))
    seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
    sleeve = np.array([("sleeve" in s or "cuff" in s) for s in seg])
    out = []
    for sign, (c, u, wrist) in arms.items():
        side = sleeve & (sign * v[:, 0] > 0)
        out.append(((v[side] - c) @ u).max() - wrist)
    return round(float(np.mean(out)), 1)


def main(src="aibf16_3090_v2_g03_blue_hoodie", ruffle="1"):
    dm.ensure_system_json()
    body_yaml = dm.GC_ROOT / "assets" / "bodies" / "mean_male.yaml"
    bv, bf = igl.read_triangle_mesh(str(body_yaml.with_suffix(".obj")))
    bv *= 100
    arms = {s: arm(bv, s) for s in (1, -1)}

    folder = dm.OUT_ROOT / src
    design = yaml.safe_load((folder / "design_used.yaml").read_text(encoding="utf-8"))["design"]
    design["sleeve"]["cuff"]["type"]["v"] = "CuffBand"  # 羅紋縮口；長度用預設 cuff_len 0.1（約 5.7 cm）
    design["sleeve"]["cuff"]["top_ruffle"]["v"] = float(ruffle)
    name = src.replace("_v2_", "_cuff_").replace("aibf16_3090", "aibf16") + ("" if float(ruffle) == 1 else f"_r{ruffle}")
    out, _ = build(design, name, body_yaml, (bv, bf))
    (out / "design_used.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True), encoding="utf-8")
    result = {"原本（沒有縮口）袖口離手腕 cm": cuff_end(folder, src, arms),
              "加縮口後 cm": cuff_end(out, name, arms), "cuff_len": design["sleeve"]["cuff"]["cuff_len"]["v"],
              "top_ruffle": float(ruffle),
              "縫合": SEW}
    (out / "cuff_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(name, result)


if __name__ == "__main__":
    main(*sys.argv[1:])
