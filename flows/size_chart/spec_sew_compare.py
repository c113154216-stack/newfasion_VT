# -*- coding: utf-8 -*-
r"""把 spec_to_pattern.py 照尺寸表打好的版片縫到 SMPL-X 中性身體上，跟商品照並排。

同一件（預設 z06 長袖 T 的 M 號）縫三種肩寬，看「把肩線往外推」會不會讓袖籠縫壞：
  不推（GarmentCode 原本，約 34.5 cm）、推到 55 cm、推到最大（約 59 cm，尺寸表 62）
胸寬、衣長、袖長三種都照尺寸表。縫合設定同 export_neutral（沒有重力）。

跑法（在 newfashion_VT 根目錄，先跑過 spec_to_pattern.py）：
    .venv\Scripts\python flows\size_chart\spec_sew_compare.py [網路衣服資料夾] [尺碼]
輸出：output/spec_to_pattern/<資料夾>/縫合_<尺碼>.png、各版本的 _sewn.obj
"""
import json
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
BODY_FIT = HERE.parent / "body_fit"  # 共用的打版、縫合、匯出程式（照身體生成那條流程）
sys.path.insert(0, str(BODY_FIT))
sys.path.insert(0, str(BODY_FIT / "experiments"))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
from assets.bodies.body_params import BodyParameters  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import build  # noqa: E402
from body_compare_figure import SKIN, shade, plt  # noqa: E402

ROOT = sp.ROOT


def shoulder_for(design, target_cm):
    """找「打版用肩寬」讓版片肩寬＝target_cm（量到的肩寬大約隨它 1:1 變）。"""
    sw = sp.BASE_BODY.params["shoulder_w"]
    for _ in range(6):
        m = sp.measure(sp.build(design, {"shoulder_w": sw}))
        if abs(m["shoulder"] - target_cm) < 0.2:
            break
        sw += target_cm - m["shoulder"]
    return sw, m


def main(folder=None, size="M"):
    folder = Path(folder) if folder else HERE / "網路衣服_尺寸資料" / "z06_長袖T"
    if not folder.is_absolute():
        folder = ROOT / folder
    out = ROOT / "output" / "spec_to_pattern" / folder.name
    res = json.loads((out / "結果.json").read_text(encoding="utf-8"))[size]
    design = yaml.safe_load((out / f"design_{size}.yaml").read_text(encoding="utf-8"))["design"]

    sp.dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    sp.BASE_BODY = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body_mesh = (bv * 100, bf)
    base = yaml.safe_load(Path(body_yaml).read_text())

    variants = [("不推肩線", sp.BASE_BODY.params["shoulder_w"]),
                ("肩寬推到 55", shoulder_for(design, 55.0)[0]),
                ("肩寬推到最大", res["參數"]["打版用肩寬 shoulder_w"])]
    sewn = []
    for label, sw in variants:
        m = sp.measure(sp.build(design, {"shoulder_w": sw}))
        body = json.loads(json.dumps(base))
        body["body"]["shoulder_w"] = float(sw)
        yml = out / f"body_shoulder_{sw:.1f}.yaml"
        yml.write_text(yaml.safe_dump(body), encoding="utf-8")
        name = f"spec_{folder.name.split('_')[0]}_{size}_sw{sw:.0f}"
        print(f"[{label}] 打版用肩寬 {sw:.1f} → 版片肩寬 {m['shoulder']:.1f}，縫合中…", flush=True)
        _, x, f, _, _ = build(design, name, yml, body_mesh)
        sewn.append((label, m, x, f))

    # 圖：商品照正、背 ＋ 三種版本正、背
    import matplotlib.image as mpimg
    keep = bv[bf].mean(1)[:, 1] * 100 > 60  # 只畫腰以上，比較清楚
    b = (body_mesh[0], bf[keep])
    flip = np.array([-1.0, 1.0, -1.0])
    fig, axes = plt.subplots(2, 1 + len(sewn), figsize=(4.2 * (1 + len(sewn)), 9))
    for r, side in enumerate(["front", "back"]):
        ax = axes[r, 0]
        ax.axis("off")
        img = folder / f"{folder.name}_{side}.jpg"
        if img.exists():
            ax.imshow(mpimg.imread(str(img)))
            ax.set_title(("商品照（正面）" if side == "front" else "商品照（背面）"), fontsize=12)
        for c, (label, m, x, f) in enumerate(sewn, start=1):
            ax = axes[r, c]
            s = flip if side == "back" else 1.0
            shade(ax, [(b[0] * s, b[1], SKIN), (x * s, f, "#e8e8e8")])
            ax.set_xlim(-95, 95), ax.set_ylim(55, 175)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{label}\n肩寬 {m['shoulder']:.1f}／胸寬 {m['chest']:.1f}／衣長 {m['length']:.1f}／袖長 {m['sleeve']:.1f}",
                             fontsize=10)
    t = res["目標"]
    fig.suptitle(f"{folder.name} {size} 號：照尺寸表打版後縫合（尺寸表：肩寬 {t.get('shoulder')}、胸寬 {t['chest']}、"
                 f"衣長 {t['length']}、袖長 {t['sleeve']}；版片上量，單位 cm）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = out / f"縫合_{size}.png"
    fig.savefig(png, dpi=110)
    print("輸出：", png)


if __name__ == "__main__":
    main(*sys.argv[1:])
