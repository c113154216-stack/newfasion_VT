# -*- coding: utf-8 -*-
"""周誌用：對照圖裡後片中間「凸起來」的是什麼（z06 長袖 T，M，_batch 結果，2026-11-25）。
① 從側面切開（身體中線 x＝0 的剖面）：後領口實際比肩頸點高、離脖子背面很遠
② 從正後方看 vs ③ 從背後斜上方 30° 看（對照圖用的角度）：同一件衣服，凸起看起來差一倍
跑法（newfashion_VT 根目錄）：.venv\\Scripts\\python docs\\驗證_尺寸\\fig_back_neck.py
"""
import json
from pathlib import Path

import igl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "output" / "spec_to_pattern" / "_batch" / "z06" / "M"
OUT = ROOT / "docs" / "images" / "size_chart" / "周誌11_後領口.png"
x, f = igl.read_triangle_mesh(str(D / "z06_M.obj"))
seg = np.array((D / "segmentation.txt").read_text().splitlines())
lm = json.loads((D / "量測.json").read_text(encoding="utf-8"))["量測點"]
bv, bf = igl.read_triangle_mesh(str(ROOT / "output" / "physics_export" / "_bodies" / "smplx_neutral_theta0_feet0.obj"))
bv = bv * 100  # 縫合時身體是公分


def slice_x0(v, f, keep=None):
    """網格和 x＝0 平面的交線（線段）。keep：只取這些面。"""
    f = f if keep is None else f[keep]
    s = np.sign(v[f, 0])
    segs = []
    for tri, sg in zip(f, s):
        if abs(sg.sum()) == 3:
            continue
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = v[tri[a]], v[tri[b]]
            if (pa[0] <= 0) != (pb[0] <= 0):
                t = pa[0] / (pa[0] - pb[0])
                q = pa + t * (pb - pa)
                pts.append([q[2], q[1]])  # 橫軸 z（＋＝胸前），縱軸 y
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)


back = np.char.find(seg, "btorso") >= 0
front = np.char.find(seg, "ftorso") >= 0
fb = np.any(back[f], axis=1)
ff = np.any(front[f], axis=1) & ~fb
c = np.flatnonzero(back & (np.abs(x[:, 0]) < 2))
top = c[x[c, 1].argmax()]                       # 後中心領口最高點
neck_y = np.mean([x[v, 1] for v in lm["肩頸點"].values()])
sb = bv[(np.abs(bv[:, 0]) < 1.5) & (np.abs(bv[:, 1] - x[top, 1]) < 1.0)]
body_back_z = sb[:, 2].min()                    # 同高度脖子背面
gap = body_back_z - x[top, 2]
rise = x[top, 1] - neck_y
print("後中心領口", np.round(x[top], 1), "肩頸點高", round(neck_y, 1), "高", round(rise, 1), "離脖子", round(gap, 1))

fig = plt.figure(figsize=(18, 8.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])

# ① 側面剖面
ax = fig.add_subplot(gs[0])
ax.add_collection(LineCollection(slice_x0(bv, bf), colors="#b08b6e", lw=1.6))
ax.add_collection(LineCollection(slice_x0(x, f, fb), colors="#2f7fc1", lw=2.6))
ax.add_collection(LineCollection(slice_x0(x, f, ff), colors="#8e6cc9", lw=2.6))
ax.axhline(neck_y, c="green", ls="--", lw=1.3)
ax.text(17, neck_y + 0.4, f"肩頸點的高度（{neck_y:.1f}）", color="green", fontsize=11, ha="right", va="bottom")
ax.scatter([x[top, 2]], [x[top, 1]], s=120, c="red", ec="k", zorder=5)
ax.annotate("", xy=(x[top, 2], neck_y), xytext=(x[top, 2], x[top, 1]), arrowprops=dict(arrowstyle="<->", color="red", lw=1.8))
ax.text(x[top, 2] - 0.8, (neck_y + x[top, 1]) / 2, f"高 {rise:.1f} cm", color="red", fontsize=12, ha="right", va="center")
ax.annotate("", xy=(body_back_z, x[top, 1]), xytext=(x[top, 2], x[top, 1]), arrowprops=dict(arrowstyle="<->", color="red", lw=1.8))
ax.text((body_back_z + x[top, 2]) / 2, x[top, 1] + 0.9, f"離脖子 {gap:.1f} cm", color="red", fontsize=12, ha="center", va="bottom")
real = (body_back_z - 0.4, neck_y - 1.8)
ax.scatter([real[0]], [real[1]], s=110, marker="*", c="gold", ec="k", zorder=5)
ax.annotate("真的 T 恤（示意）：後領貼著脖子根部，\n比肩頸點低 1.5～2 cm", xy=real, xytext=(4, 133), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color="#8a6d00"), color="#8a6d00", ha="center",
            bbox=dict(fc="white", ec="#8a6d00", alpha=0.9))
ax.text(-23, 163, "← 背後", fontsize=12)
ax.text(10, 163, "胸前 →", fontsize=12)
ax.set_xlim(-24, 18)
ax.set_ylim(126, 166)
ax.set_aspect("equal")
ax.set_title("① 從側面切開（身體正中間的剖面）", fontsize=13.5)
ax.legend(handles=[plt.Line2D([], [], c="#b08b6e", lw=2, label="身體"), plt.Line2D([], [], c="#2f7fc1", lw=3, label="衣服後片"),
                   plt.Line2D([], [], c="#8e6cc9", lw=3, label="衣服前片"),
                   plt.Line2D([], [], marker="o", ls="", mfc="red", mec="k", ms=10, label="後中心領口")],
          loc="lower left", fontsize=10.5)
ax.set_xlabel("前後（cm）")
ax.set_ylabel("高度（cm）")

# ②③ 兩個角度
PAL = {"btorso": "#9ecae1", "ftorso": "#cbbbe6", "sleeve": "#fdd0a2"}
fc = []
for t in f:
    labs = [seg[v] for v in t if not seg[v].startswith("stitch")]
    fc.append(next((cc for k, cc in PAL.items() if labs and k in labs[0]), "#dddddd"))
fc = np.array(fc)
for k, (deg, title) in enumerate(((0, "② 從正後方看"), (30, "③ 從背後斜上方 30° 看（對照圖用的角度）"))):
    a = np.radians(deg)

    def proj(p):
        return np.stack([-p[..., 0], p[..., 1] * np.cos(a) - p[..., 2] * np.sin(a)], axis=-1)

    dep = (-x[f][:, :, 2] * np.cos(a) + x[f][:, :, 1] * np.sin(a)).mean(1)
    o = np.argsort(dep)
    ax = fig.add_subplot(gs[1 + k])
    ax.add_collection(PolyCollection(proj(x[f])[o], facecolors=fc[o], edgecolors="#00000014", linewidths=0.2))
    pt, pn = proj(x[top]), [proj(x[v]) for v in lm["肩頸點"].values()]
    ny = np.mean([q[1] for q in pn])
    for q in pn:
        ax.scatter([q[0]], [q[1]], s=80, c="green", ec="k", zorder=5)
    ax.scatter([pt[0]], [pt[1]], s=110, c="red", ec="k", zorder=5)
    ax.plot([-14, 14], [ny, ny], c="green", ls="--", lw=1.2)
    ax.annotate("", xy=(pt[0] + 6, ny), xytext=(pt[0] + 6, pt[1]), arrowprops=dict(arrowstyle="<->", color="red", lw=1.8))
    ax.text(pt[0] + 7, (ny + pt[1]) / 2, f"畫面上高 {pt[1] - ny:.1f} cm", color="red", fontsize=12, va="center")
    top_s = proj(x)[:, 1].max()
    ax.set_xlim(-30, 30)
    ax.set_ylim(top_s - 30, top_s + 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13.5)
    if k == 1:
        ax.text(0.5, 0.04, f"往後跑的 {gap:.1f} cm 被斜著投影成「往上」，\n所以看起來比實際（{rise:.1f}）凸很多",
                transform=ax.transAxes, ha="center", fontsize=11.5, bbox=dict(fc="white", ec="#999"))
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="red", mec="k", ms=10, label="後中心領口"),
                    plt.Line2D([], [], marker="o", ls="", mfc="green", mec="k", ms=9, label="肩頸點（左右）")],
           loc="lower right", ncol=2, fontsize=11, bbox_to_anchor=(0.98, 0.02))
fig.suptitle("對照圖後片中間凸起來的是「後領口」：一半是視角，一半是領口真的往上、往後跑（z06 長袖 T，M）", fontsize=14.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(OUT, dpi=105, bbox_inches="tight")
print(OUT)
