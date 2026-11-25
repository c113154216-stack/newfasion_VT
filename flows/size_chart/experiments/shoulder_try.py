# -*- coding: utf-8 -*-
r"""試驗：落肩（肩寬照尺寸表、比身體肩膀寬）為什麼縫壞、怎麼救。

spec_sew_compare.py 發現：打版用的身體肩寬加大後，袖子縫好會掉在手臂下面。猜兩個原因：
  1. 袖子的休息角度（sleeve_angle，i10 是 36°）讓袖片一開始往下斜，肩點又被推到 28 cm 外，袖子起點在手臂下方，手臂接不住
  2. 肩線照身體肩斜（約 22°）一路往外斜下去，推到手臂上方時，衣身的肩點已經低於手臂
各做一件（z06 長袖 T 的 M 號，款式用 bf16_z 自己的推論）：
  A 只推肩寬            B 推肩寬＋袖子角度 0（實際取肩斜）      C 再把打版用的肩斜改成 10°
跑法（在 repo 根目錄）：
    .venv\Scripts\python flows\size_chart\experiments\shoulder_try.py [目標肩寬，預設 50] [C 的肩斜，預設 10]
輸出：output/spec_to_pattern/shoulder_try/肩寬<目標>.png
"""
import copy
import json
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "body_fit"))
sys.path.insert(0, str(HERE.parents[1] / "body_fit" / "experiments"))
import spec_batch as sb  # noqa: E402
import spec_to_pattern as sp  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import build  # noqa: E402
from body_compare_figure import SKIN, shade, plt  # noqa: E402

OUT = sp.ROOT / "output" / "spec_to_pattern" / "shoulder_try"


def main(target_shoulder="50", incl="10"):
    target_shoulder, incl = float(target_shoulder), float(incl)
    OUT.mkdir(parents=True, exist_ok=True)
    sp.dm.ensure_system_json()
    body_obj, body_yaml, _ = prepare_smplx_body("neutral")
    sp.BASE_BODY = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body_mesh = (bv * 100, bf)
    base_yaml = yaml.safe_load(Path(body_yaml).read_text())

    gid = "z06_長袖T"
    info = json.loads((sb.DATA / gid / "尺寸.json").read_text(encoding="utf-8"))
    vals = info["sizes"]["M"]
    table = {sp.KEYS[c]: float(v) for c, v in zip(info["cols"], vals) if c in sp.KEYS}
    target = {k: table[k] for k in ("chest", "length", "sleeve")}
    template = sb.template_for(gid, sp.BASE_BODY)
    design, _, _ = sb.fit2(target, template)

    variants = [("A 只推肩寬", {}, None), ("B ＋袖子角度 0", {}, 0), (f"C ＋肩斜 {incl:g}°", {"shoulder_incl": incl}, 0)]
    rows = []
    for label, body_extra, angle in variants:
        d = copy.deepcopy(design)
        if angle is not None:
            d["sleeve"]["sleeve_angle"]["v"] = angle
        # 找打版用肩寬：版片肩寬＝目標
        sw = sp.BASE_BODY.params["shoulder_w"]
        for _ in range(6):
            try:
                m = sp.measure(sp.build(d, {"shoulder_w": sw, **body_extra}))
            except AssertionError:
                sw -= 1
                continue
            if abs(m["shoulder"] - target_shoulder) < 0.3:
                break
            sw += target_shoulder - m["shoulder"]
        body = copy.deepcopy(base_yaml)
        body["body"]["shoulder_w"] = float(sw)
        body["body"].update(body_extra)
        yml = OUT / f"body_{label[0]}.yaml"
        yml.write_text(yaml.safe_dump(body), encoding="utf-8")
        print(f"[{label}] 打版用肩寬 {sw:.1f} → 版片肩寬 {m['shoulder']:.1f}，縫合中…", flush=True)
        _, x, f, _, _ = build(d, f"shtry_{label[0]}_{target_shoulder:.0f}_{incl:g}", yml, body_mesh)
        rows.append((label, m, x, f))

    keep = bv[bf].mean(1)[:, 1] * 100 > 60
    fig, axes = plt.subplots(2, len(rows), figsize=(4.6 * len(rows), 9))
    flip = np.array([-1.0, 1.0, -1.0])
    for c, (label, m, x, f) in enumerate(rows):
        for r, s in enumerate([1.0, flip]):
            ax = axes[r, c]
            shade(ax, [(body_mesh[0] * s, bf[keep], SKIN), (x * s, f, "#e8e8e8")])
            ax.set_xlim(-95, 95), ax.set_ylim(55, 175)
            ax.axis("off")
        axes[0, c].set_title(f"{label}\n版片肩寬 {m['shoulder']:.1f}（尺寸表 {table['shoulder']}）", fontsize=11)
    fig.suptitle(f"z06 長袖 T M 號：肩寬推到 {target_shoulder:.0f} cm 的三種做法（上：正面，下：背面）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = OUT / f"shoulder_{target_shoulder:.0f}_incl{incl:g}.png"
    fig.savefig(png, dpi=110)
    print("輸出：", png)


if __name__ == "__main__":
    main(*sys.argv[1:])
