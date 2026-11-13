# -*- coding: utf-8 -*-
"""
袖長規則前後對比（2026-11-09）：
  1. 袖長規則_前後_5件.png：5 件長袖，規則前（AI 參數）和規則後並排，正面看右手臂，標出手腕位置
  2. 袖口離手腕_全部.png：全部衣服的袖口位置。長袖：規則前→後離手腕多少 cm（理想 0）；
     短袖、無袖沒有目標位置（照 AI），只畫袖口在手臂上的位置（肩關節到手腕的 %）

直接量匯出的 obj（SMPL-X 原始座標、公尺）：衣服頂點沿手臂方向（肩關節 → 手腕）最遠的位置減掉手腕，左右平均。
長袖時最遠的點就是袖口（和 sleevelen.reach 相同）；短袖是袖口；無袖是肩膀附近。
規則前＝$TEMP/neutral_before_sleevecomp（AI 袖長參數、同樣的衣長校正），規則後＝output/physics_export/neutral。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python flows\\body_fit\\experiments\\sleeve_rule_compare.py
"""
import json
import os
import sys
from pathlib import Path

import igl
import yaml
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from body_compare_figure import SKIN, shade, plt  # noqa: E402
import sleevelen  # noqa: E402

ROOT = HERE.parents[2]
AFTER = ROOT / "output" / "physics_export" / "neutral"
BEFORE = Path(os.environ["TEMP"]) / "neutral_before_sleevecomp"
BODY = ROOT / "output" / "physics_export" / "_bodies" / "smplx_neutral_theta0.obj"
DST = ROOT / "output" / "chatgarment"
PICK = 5
BEF, AFT, GRAY = "#d9822b", "#2f7fc1", "#9a9a9a"


def cuff(v, arm):
    """(離手腕 cm 左右平均, 在手臂上的位置 %)"""
    gaps, pct = [], []
    for sign, (c, u, wrist) in arm.items():
        side = v[sign * v[:, 0] > 0]
        far = float(((side - c) @ u).max())
        gaps.append(far - wrist)
        pct.append(100 * far / wrist)
    return float(np.mean(gaps)), float(np.mean(pct))


def main():
    arm = sleevelen.arms(0.0)  # 原始座標（腳底不在 0）
    import smplx_body
    J = smplx_body.load_model("neutral")["J_regressor"] @ smplx_body.make_body("neutral")[0] * 100
    c, u, wrist = arm[1 if J[18, 0] > 0 else -1]
    elbow_pct = 100 * float((J[18] - c) @ u) / wrist  # 手肘在手臂上的位置 %
    bv, bf = igl.read_triangle_mesh(str(BODY))
    bv *= 100
    rows = []
    for d in sorted(p for p in AFTER.iterdir() if p.is_dir() and not p.name.startswith("_")):
        g = d.name
        fx = json.loads((d / "fix.json").read_text(encoding="utf-8"))
        design = json.dumps(fx, ensure_ascii=False)
        va, _ = igl.read_triangle_mesh(str(d / f"{g}.obj"))
        vb, _ = igl.read_triangle_mesh(str(BEFORE / g / f"{g}.obj"))
        a, pa = cuff(va * 100, arm)
        b, pb = cuff(vb * 100, arm)
        long = isinstance(fx.get("袖長規則"), dict)

        sv = yaml.safe_load((d / "design_used.yaml").read_text(encoding="utf-8"))["design"]["sleeve"]
        kind = "長袖" if long else ("無袖" if sv["sleeveless"]["v"] else "短袖")
        rows.append(dict(g=g, kind=kind, before=b, after=a, pct_after=pa, pct_before=pb))
    (ROOT / "output" / "slide_fix" / "sleeve_rule_compare.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    longs = [r for r in rows if r["kind"] == "長袖"]
    print(f"長袖 {len(longs)}：前 平均 {np.mean([r['before'] for r in longs]):.1f}，後 平均 "
          f"{np.mean([r['after'] for r in longs]):.1f}、最大 |差| {max(abs(r['after']) for r in longs):.1f}")

    # ---- 圖一：挑 5 件（規則前最短、中間、接近 0 都有）----
    s = sorted(longs, key=lambda r: r["before"])
    idx = np.unique(np.linspace(0, len(s) - 1, PICK).round().astype(int))
    pick = [s[i] for i in idx]
    c, u, wrist = arm[-1]  # 右手（x 負）
    wpos = c + u * wrist
    fig, axes = plt.subplots(len(pick), 2, figsize=(14, 4.2 * len(pick)))
    for row, r in zip(axes, pick):
        for ax, (src, col, lab, val) in zip(row, ((BEFORE, BEF, "規則前（AI 參數）", r["before"]),
                                                   (AFTER, AFT, "規則後", r["after"]))):
            v, f = igl.read_triangle_mesh(str(src / r["g"] / f"{r['g']}.obj"))
            shade(ax, [(bv, bf, SKIN), (v * 100, f, col)])
            ax.set_xlim(-95, 5)
            ax.set_ylim(wpos[1] - 22, wpos[1] + 30)
            ax.axvline(wpos[0], color="k", ls="--", lw=1.2)
            ax.text(wpos[0], wpos[1] + 27, " 手腕", fontsize=11, va="top")
            ax.set_title(f"{r['g']}　{lab}　袖口離手腕 {val:+.1f} cm", fontsize=12, color=col)
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("袖長規則前後（正面看右手臂，SMPL-X 中性 T-pose）\n虛線＝手腕；負＝沒到手腕、正＝超過",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(DST / "袖長規則_前後_5件.png", dpi=70)

    # ---- 圖二：全部 ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 11), gridspec_kw={"width_ratios": [1.15, 1]})
    s = sorted(longs, key=lambda r: r["before"])
    y = np.arange(len(s))
    for i, r in enumerate(s):
        a1.plot([r["before"], r["after"]], [i, i], color=GRAY, lw=1, zorder=1)
    a1.scatter([r["before"] for r in s], y, color=BEF, s=40, label="規則前（AI 參數）", zorder=2)
    a1.scatter([r["after"] for r in s], y, color=AFT, s=40, label="規則後", zorder=3)
    a1.axvspan(-1, 1, color=AFT, alpha=0.08)
    a1.axvline(0, color="k", lw=1)
    a1.set_yticks(y, [r["g"] for r in s], fontsize=9)
    a1.set_xlabel("袖口離手腕 cm（0＝剛好到手腕；淺藍＝±1 cm 容許）", fontsize=12)
    a1.set_title(f"長袖 {len(s)} 件：理想位置是手腕\n規則前 平均 {np.mean([r['before'] for r in s]):+.1f} cm"
                 f" → 規則後 平均 {np.mean([r['after'] for r in s]):+.1f} cm", fontsize=14)
    a1.legend(loc="lower right", fontsize=11)
    a1.grid(axis="x", alpha=0.3)

    o = sorted([r for r in rows if r["kind"] != "長袖"], key=lambda r: r["pct_after"])
    y = np.arange(len(o))
    a2.barh(y, [r["pct_after"] for r in o], color=[GRAY if r["kind"] == "無袖" else "#7aa874" for r in o])
    for x, lab in ((0, "肩關節"), (elbow_pct, "手肘"), (100, "手腕")):
        a2.axvline(x, color="k", ls=":", lw=1)
        a2.text(x, len(o) + 0.3, lab, ha="center", fontsize=11)
    a2.set_yticks(y, [f"{r['g']}（{r['kind']}）" for r in o], fontsize=9)
    a2.set_xlim(-5, 105)
    a2.set_ylim(-0.7, len(o) + 1.2)
    a2.set_xlabel("袖口在手臂上的位置（肩關節＝0%，手腕＝100%）", fontsize=12)
    a2.set_title(f"短袖、無袖 {len(o)} 件：沒有理想位置（照 AI 參數、規則沒動）", fontsize=14)
    a2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DST / "袖口離手腕_全部.png", dpi=65)
    print("ok", [r["g"] for r in pick])


if __name__ == "__main__":
    main()
