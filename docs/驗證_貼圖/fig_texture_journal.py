# -*- coding: utf-8 -*-
"""周誌用：縫線貼圖、扣子（測試中，還沒進正式流程）的對照圖（2026-11-25）。
  周誌11_縫線貼圖.png   z06 M：沒貼圖 vs 縫線貼圖（整件＋放大肩膀、袖口），加上貼圖本身
  周誌11_扣子AB.png     z02 M 襯衫、z01 M 帽 T：A 全畫在貼圖上 vs B 貼圖＋立體零件
  周誌11_下襬AB.png     z01 M 帽 T 下襬：反摺車縫 vs 羅紋
資料：output/seam_trim_demo、output/detail_demo（tools/texture/seam_trim.py、detail_trim.py 的輸出）。
跑法（newfashion_VT 根目錄）：.venv\\Scripts\\python docs\\驗證_貼圖\\fig_texture_journal.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from render_textured import Camera, render  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
ROOT = Path(__file__).resolve().parents[2]
SEAM = ROOT / "output" / "seam_trim_demo"
DET = ROOT / "output" / "detail_demo"
OUT = ROOT / "docs" / "images" / "size_chart"
LIGHT = (0.93, 0.93, 0.92)  # 淺色布（看得清楚）


def show(ax, img, title, colr="k"):
    ax.imshow(img)
    ax.set_title(title, fontsize=13, color=colr)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#ccc")


# ───────── 縫線貼圖：z06 M ─────────
g = SEAM / "z06_M"
whole = Camera(yaw=18, pitch=8, window=(-0.82, 0.82, 0.76, 1.58), px_per_m=520)
shoulder = Camera(yaw=35, pitch=10, window=(0.02, 0.38, 1.22, 1.50), px_per_m=1800)
cuff = Camera(yaw=10, pitch=5, window=(0.48, 0.84, 1.18, 1.40), px_per_m=1800)
imgs = {}
for key, cp, npng in (("plain", "plain_color.png", None), ("seam", "seam_color.png", "seam_normal.png")):
    for vk, cam in (("whole", whole), ("shoulder", shoulder), ("cuff", cuff)):
        imgs[key, vk] = render(g, cp, npng, cam, tint=LIGHT)
        print("縫線", key, vk)
fig = plt.figure(figsize=(19, 12.5))
gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1.05], hspace=0.12, wspace=0.05)
for r, (key, title, colr) in enumerate((("plain", "沒貼圖（現在試穿頁的樣子）", "#555"), ("seam", "縫線貼圖", "#c0561f"))):
    show(fig.add_subplot(gs[r, 0]), imgs[key, "whole"], f"{title}：整件", colr)
    show(fig.add_subplot(gs[r, 1]), imgs[key, "shoulder"], f"{title}：放大肩膀、袖籠", colr)
    show(fig.add_subplot(gs[r, 2]), imgs[key, "cuff"], f"{title}：放大袖口", colr)
uv = np.asarray(Image.open(g / "seam_color.png").convert("RGB"))
ax = fig.add_subplot(gs[0, 3])
show(ax, uv[:1024, 300:1024], "貼圖本身（UV，前片＋後片的一部分）")
ax = fig.add_subplot(gs[1, 3])
ax.axis("off")
ax.text(0, 1, "畫了什麼", fontsize=14, weight="bold", va="top")
ax.text(0, 0.9, "• 每條縫線：往下陷一條細縫，\n   顏色深一點\n"
                "• 袖籠、肩線、領口、袖口：\n   旁邊 0.15 cm 一條車縫線\n"
                "• 圓領：一圈 2 cm 羅紋帶\n"
                "• 下襬、袖口：反摺，\n   離邊 2.0、2.6 cm 雙針車縫\n\n"
                "網格、頂點數完全沒變，\n只多一張顏色貼圖和一張法線貼圖\n（假的凹凸）",
        fontsize=12, va="top", linespacing=1.55)
fig.suptitle("縫線貼圖（z06 長袖 T，M）：上排沒貼圖、下排加縫線貼圖，網格一模一樣", fontsize=15, y=0.95)
fig.savefig(OUT / "周誌11_縫線貼圖.png", dpi=95, bbox_inches="tight")
plt.close(fig)

# ───────── 扣子：A vs B ─────────
cams = {
    "z02_M": [Camera(yaw=12, pitch=4, window=(-0.30, 0.30, 0.96, 1.50), px_per_m=1100),
              Camera(yaw=40, pitch=6, window=(-0.12, 0.10, 1.20, 1.40), px_per_m=2600)],
    "z01_M": [Camera(yaw=12, pitch=4, window=(-0.34, 0.34, 0.86, 1.62), px_per_m=950),
              Camera(yaw=40, pitch=6, window=(-0.12, 0.16, 1.18, 1.44), px_per_m=2300)],
}
names = {"z02_M": "z02 襯衫 M（門襟、6 顆扣子）", "z01_M": "z01 帽 T M（口袋、帽繩）"}
fig, axs = plt.subplots(2, 4, figsize=(20, 12.5), gridspec_kw=dict(hspace=0.16, wspace=0.04))
for r, gid in enumerate(("z02_M", "z01_M")):
    d = DET / gid
    for j, (lab, cp, npng, parts, colr) in enumerate((("A：全部畫在貼圖上", "detail_color.png", "detail_normal.png", False, "#2a6db5"),
                                                      ("B：貼圖＋立體零件", "seam_color.png", "seam_normal.png", True, "#c0561f"))):
        for c, cam in enumerate(cams[gid]):
            img = render(d, cp, npng, cam, parts=parts)
            show(axs[r, j * 2 + c], img, f"{lab}{'（放大）' if c else ''}", colr)
            print("扣子", gid, lab, c)
    axs[r, 0].set_ylabel(names[gid], fontsize=13)
axs[1, 1].text(0.5, 0.5, "A 沒有帽繩：\n帽繩是垂下來的立體繩子，\n畫在貼圖上會黏在衣服上，\n所以只有 B 做", transform=axs[1, 1].transAxes,
               ha="center", va="center", fontsize=13, color="white", linespacing=1.6)
fig.suptitle("扣子、口袋、帽繩：A（畫在貼圖上）vs B（立體零件）。衣服網格兩邊一模一樣", fontsize=15, y=0.93)
fig.savefig(OUT / "周誌11_扣子AB.png", dpi=95, bbox_inches="tight")
plt.close(fig)

# ───────── 帽 T 下襬：反摺 vs 羅紋 ─────────
cam = Camera(yaw=15, pitch=-4, window=(-0.36, 0.36, 0.84, 1.10), px_per_m=1500)
fig, axs = plt.subplots(1, 2, figsize=(17, 4.6), gridspec_kw=dict(wspace=0.04))
for ax, (gid, lab) in zip(axs, (("z01_M", "反摺：離邊 2.0、2.6 cm 雙針車縫（只改貼圖）"), ("z01_M_rib", "羅紋：多一片下襬羅紋版片（網格要重縫）"))):
    show(ax, render(DET / gid, "detail_color.png", "detail_normal.png", cam), lab)
fig.suptitle("z01 帽 T（M）下襬的兩種做法", fontsize=15, y=0.98)
fig.savefig(OUT / "周誌11_下襬AB.png", dpi=95, bbox_inches="tight")
plt.close(fig)
print("完成")
