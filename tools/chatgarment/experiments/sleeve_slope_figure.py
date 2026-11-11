# -*- coding: utf-8 -*-
"""
袖長斜率的驗證圖（2026-11-10）：讀 output/physics_export/neutral/*/fix.json 的「袖長規則」
  左：32 件長袖「AI 袖長參數」vs「用 AI 參數縫好的袖口離手腕」，最小平方法直線（跨衣服的斜率）
  右：每件衣服自己的斜率（第一次調參數後，袖口實際移動 cm ÷ 參數改變量），和規則用的 0.9 × 54 比較
輸出 output/chatgarment/袖長斜率_驗證.png、docs/images/ 同名，並印出計算過程

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\sleeve_slope_figure.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from body_compare_figure import plt  # noqa: E402  (含中文字型設定)

ROOT = HERE.parents[2]
SRC = ROOT / "output" / "physics_export" / "neutral"
ARM = 53.9697  # mean_all.yaml 的 arm_length
RULE = 0.9 * ARM


def main():
    rows = []
    for p in sorted(SRC.glob("*/fix.json")):
        s = json.loads(p.read_text(encoding="utf-8")).get("袖長規則")
        if isinstance(s, dict):
            L, cm = s["袖長參數"], [s["AI 參數縫好 cm"]] + s["縫好的袖口離手腕 cm（每一輪）"]
            rows.append(dict(g=p.parent.name, L=L, cm=cm))
    x = np.array([r["L"][0] for r in rows])
    y = np.array([r["cm"][0] for r in rows])
    n, xm, ym = len(x), x.mean(), y.mean()
    sxy, sxx, syy = ((x - xm) * (y - ym)).sum(), ((x - xm) ** 2).sum(), ((y - ym) ** 2).sum()
    a = sxy / sxx
    b = ym - a * xm
    r = sxy / np.sqrt(sxx * syy)
    print(f"n={n}  x̄={xm:.4f}  ȳ={ym:.3f}")
    print(f"Sxy=Σ(x-x̄)(y-ȳ)={sxy:.4f}  Sxx=Σ(x-x̄)²={sxx:.5f}  Syy={syy:.3f}")
    print(f"斜率 a=Sxy/Sxx={a:.2f}  截距 b=ȳ-a·x̄={b:.2f}  r=Sxy/√(Sxx·Syy)={r:.3f}")
    print(f"x=1 時 y={a + b:.2f}；a/ARM={a / ARM:.3f}；規則斜率 0.9×{ARM:.2f}={RULE:.2f}")
    # 每件自己的斜率：第一次調整
    own = np.array([(r["cm"][1] - r["cm"][0]) / (r["L"][1] - r["L"][0]) for r in rows])
    print(f"每件斜率（第一次調整）：中位數 {np.median(own):.1f}、範圍 {own.min():.1f}～{own.max():.1f}")
    for rr, o in sorted(zip(rows, own), key=lambda t: t[1]):
        print(f"  {rr['g']:<34} 參數 {rr['L']}  袖口 {rr['cm']}  斜率 {o:.1f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7.5))
    a1.scatter(x, y, s=45, color="#d9822b", zorder=3)
    xs = np.linspace(0.75, 1.02, 50)
    a1.plot(xs, a * xs + b, color="#2f7fc1", lw=2, label=f"最小平方法：y ＝ {a:.1f} x {b:+.1f}（r ＝ {r:.2f}）")
    a1.plot(xs, RULE * (xs - 1) + (a + b), color="k", ls="--", lw=1.2,
            label=f"規則用的斜率 0.9 × {ARM:.0f} ＝ {RULE:.1f}（畫成同一個 x=1 點）")
    a1.axhline(0, color="gray", lw=0.8)
    a1.set_ylim(-14.5, 1.5)
    a1.axvline(1, color="gray", lw=0.8, ls=":")
    a1.annotate(f"x ＝ 1 時 y ＝ {a + b:.1f} cm", (1, a + b), (0.85, 0.6), arrowprops=dict(arrowstyle="->"), fontsize=11)
    a1.set_xlabel("AI 給的袖長參數 sleeve.length（x）", fontsize=12)
    a1.set_ylabel("用 AI 參數縫好的袖口離手腕 cm（y；負＝沒到手腕）", fontsize=12)
    a1.set_title(f"① 跨衣服：{n} 件長袖，參數越大袖口越接近手腕", fontsize=14)
    a1.legend(fontsize=10, loc="lower right")
    a1.grid(alpha=0.3)

    order = np.argsort(own)
    a2.barh(np.arange(n), own[order], color="#7aa874")
    a2.axvline(RULE, color="k", ls="--", lw=1.2, label=f"規則用的 {RULE:.1f}")
    a2.axvline(a, color="#2f7fc1", lw=1.5, label=f"跨衣服回歸 {a:.1f}")
    a2.axvline(0.3 * ARM, color="#c0392b", ls=":", lw=1.2, label=f"修正時的下限 0.3 × {ARM:.0f} ＝ {0.3 * ARM:.1f}")
    a2.set_yticks(np.arange(n), [rows[i]["g"] for i in order], fontsize=8)
    a2.set_xlabel("這件衣服自己的斜率 cm／參數 ＝ 第一次調整後袖口移動 ÷ 參數改變量", fontsize=12)
    a2.set_title(f"② 每件衣服：中位數 {np.median(own):.1f}，比規則的 {RULE:.1f} 小 → 需要第二次修正", fontsize=14)
    a2.legend(fontsize=10, loc="lower right")
    a2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "output" / "chatgarment" / "袖長斜率_驗證.png"
    fig.savefig(out, dpi=70)
    shutil.copy(out, ROOT / "docs" / "images" / out.name)
    print("ok", out)


if __name__ == "__main__":
    main()
