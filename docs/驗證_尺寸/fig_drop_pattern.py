# -*- coding: utf-8 -*-
"""周誌用：GarmentCode Shirt 和新的落肩版型 DropShoulderShirt 的後片版片對照（z04 POLO M，2026-11-25）。
資料：GarmentCode 輸出的 specification.json。
  舊（Shirt）：output/chatgarment/_backup_1125_before_drop/specb_z04_M
  新（落肩，袖籠深 ×1.2）：output/chatgarment/specb_z04_M（_batch_drop_b12 那次）
領口那條邊是半徑幾百 cm 的圓弧（13 cm 長只彎 0.05 cm），直接畫直線。
跑法（newfashion_VT 根目錄）：.venv\\Scripts\\python docs\\驗證_尺寸\\fig_drop_pattern.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
ROOT = Path(__file__).resolve().parents[2]
CG = ROOT / "output" / "chatgarment"
OUT = ROOT / "docs" / "images" / "size_chart" / "周誌11_落肩版型.png"
CHART = {"shoulder": 60.0, "chest": 58.0}  # z04 POLO M 尺寸表


def load(folder):
    sp = json.loads((CG / folder / "specb_z04_M_specification.json").read_text(encoding="utf-8"))
    return sp["pattern"]["panels"]["right_btorso"]


def outline(panel, n=40):
    """版片外框（右半片，x ≤ 0），cubic 曲線照 GarmentCode 的相對控制點取樣。回傳點列和袖籠那一段。"""
    v = np.asarray(panel["vertices"], float)
    pts, arm = [], None
    for e in panel["edges"]:
        a, b = v[e["endpoints"][0]], v[e["endpoints"][1]]
        cur = e.get("curvature")
        if isinstance(cur, dict) and cur.get("type") == "cubic":
            d = b - a
            perp = np.array([-d[1], d[0]])
            c1, c2 = (a + s[0] * d + s[1] * perp for s in cur["params"])
            t = np.linspace(0, 1, n)[:, None]
            seg = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * b
        else:
            seg = np.array([a, b])
        if str(e.get("label", "")).endswith("armhole"):
            arm = seg
        pts.append(seg[:-1])
    return np.vstack(pts + [pts[0][:1]]), arm, v


def full(p):
    """右半片 + 鏡射的左半片（後片從背後看是左右對稱的一整片）。"""
    return p, p * [-1, 1]


old_p, old_arm, old_v = outline(load("_backup_1125_before_drop/specb_z04_M"))
new_p, new_arm, new_v = outline(load("specb_z04_M"))


def lab(ax, xy, text, xytext, colr="k", fs=11):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=colr, ha="center", va="center", zorder=9,
                arrowprops=dict(arrowstyle="->", color=colr, lw=1.2),
                bbox=dict(fc="white", ec=colr, alpha=0.93, boxstyle="round,pad=0.25"))


def width_line(ax, y, x0, x1, text, colr, dy=1.6, ls="-"):
    ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", color=colr, lw=1.6, ls=ls))
    ax.text((x0 + x1) / 2, y + dy, text, ha="center", va="bottom" if dy > 0 else "top", fontsize=11.5, color=colr,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))


fig, axs = plt.subplots(1, 2, figsize=(16, 8.6))
for ax, (p, arm, v, title, col) in zip(axs[:2], (
        (old_p, old_arm, old_v, "舊：GarmentCode 的 Shirt（後片）", "#4a90d9"),
        (new_p, new_arm, new_v, "新：落肩版型 DropShoulderShirt（後片）", "#e8743b"))):
    for q in full(p):
        ax.fill(q[:, 0], q[:, 1], fc=col, alpha=0.18, ec=col, lw=2)
    for q in full(arm):
        ax.plot(q[:, 0], q[:, 1], c="purple", lw=4, solid_capstyle="round", zorder=4)
    sh = v[3]           # 肩點（袖籠上端）
    ap = v[2]           # 腋下（袖籠下端）
    side = v[1, 0]      # 側縫
    ax.axvline(side, c="#999", ls=":", lw=1)
    ax.axvline(-side, c="#999", ls=":", lw=1)
    for sx in (1, -1):
        ax.scatter([sx * sh[0]], [sh[1]], s=120, c="limegreen", ec="k", zorder=6)
        ax.scatter([sx * ap[0]], [ap[1]], s=90, c="gold", ec="k", zorder=6)
        ax.plot([sx * CHART["shoulder"] / 2] * 2, [sh[1] - 3, sh[1] + 3], c="red", lw=2.5, zorder=5)
    width_line(ax, sh[1] + 4.5, sh[0], -sh[0], f"肩寬 {2 * abs(sh[0]):.1f}", "green")
    width_line(ax, ap[1] - 4, side, -side, f"胸寬 {2 * abs(side):.1f}", "#555", dy=-1.4)
    lab(ax, sh, "肩點", (sh[0] + 7, sh[1] - 7), "green")
    lab(ax, ap, "腋下", (ap[0] + 6, ap[1] - 9), "#8a6d00")
    mid = arm[len(arm) // 2] if len(arm) > 2 else arm.mean(0)
    lab(ax, mid, "袖籠\n（接袖子）", (mid[0] - 9, mid[1] - 1), "purple")
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(-40, 40)
    ax.set_ylim(-4, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
axs[0].text(0, 12, "肩點只能在版片寬以內\n肩寬最多 ≈ 胸寬\n推不到尺寸表的 60", ha="center", fontsize=13, color="#1f4e8c")
axs[1].text(0, 12, "袖籠改成斜直線，\n肩點可以超出側縫，\n肩寬推到 60", ha="center", fontsize=13, color="#a8461a")

# 新版片上疊舊的袖籠（虛線），看差在哪
ax = axs[1]
for q in full(old_arm):
    ax.plot(q[:, 0], q[:, 1], c="#4a90d9", lw=2, ls="--", zorder=3)
lab(ax, old_arm[len(old_arm) // 2] * [-1, 1], "舊袖籠（虛線）", (20, 44), "#2a6db5")
lab(ax, new_v[2] * [-1, 1], f"腋下低 {old_v[2, 1] - new_v[2, 1]:.1f} cm\n（袖籠加深 ×1.2）", (21, 30), "#a8461a")
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=11, mfc="limegreen", mec="k", label="肩點"),
                    plt.Line2D([], [], marker="o", ls="", ms=10, mfc="gold", mec="k", label="腋下"),
                    plt.Line2D([], [], c="purple", lw=4, label="袖籠"),
                    plt.Line2D([], [], c="red", lw=2.5, label="尺寸表肩寬的位置")],
           loc="lower center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("z04 POLO（M）後片版片：尺寸表肩寬 60、胸寬 58（肩寬比胸寬大）", fontsize=15)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(OUT)
print("舊 肩點", old_v[3], "腋下", old_v[2], " 新 肩點", new_v[3], "腋下", new_v[2])
