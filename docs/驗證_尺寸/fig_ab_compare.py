# -*- coding: utf-8 -*-
"""周誌用：落肩版型的兩種換法 (a)、(b) 對照（2026-11-25）。
(a) 先用 Shirt，肩寬推不到尺寸表（差 > 0.5）才換落肩版型；(b) 落肩款全部換。
上：z04、z06 八個尺碼的 3D 肩寬 − 尺寸表（舊做法、(a)、(b)）。
下：z06 五個尺碼縫好的樣子，(a) 一排、(b) 一排，綠線是量肩寬的路徑。
資料：output/spec_to_pattern/_batch（舊＝Shirt）、_batch_drop_b12（落肩，袖籠深 ×1.2）、_batch_drop_b12/對照表.json。
跑法（newfashion_VT 根目錄）：.venv\\Scripts\\python docs\\驗證_尺寸\\fig_ab_compare.py
"""
import json
import sys
from pathlib import Path

import igl
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flows" / "size_chart"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

from spec_measure import STEINER_K, region  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
SP = ROOT / "output" / "spec_to_pattern"
OUT = ROOT / "docs" / "images" / "size_chart" / "周誌11_ab對照.png"
rows = json.loads((SP / "_batch_drop_b12" / "對照表.json").read_text(encoding="utf-8"))
C_OLD, C_A, C_B = "#9e9e9e", "#4a90d9", "#e8743b"
PAL = {"btorso": "#9ecae1", "ftorso": "#cbbbe6", "sleeve": "#fdd0a2", "cuff": "#fdae6b", "collar": "#c7e9c0"}


def surface_path(x, ff, src, dst, k=STEINER_K):
    """spec_measure.surface_dist 同一個算法，多回傳路徑（畫圖用）。"""
    n = len(x)
    e = np.sort(np.vstack([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(3, -1).T
    t = np.arange(1, k + 1) / (k + 1)
    pts = np.vstack([x, (x[uniq[:, 0]][:, None] * (1 - t)[None, :, None] + x[uniq[:, 1]][:, None] * t[None, :, None]).reshape(-1, 3)])
    on_edge = n + np.arange(len(uniq) * k).reshape(-1, k)
    fp = np.hstack([ff, on_edge[inv[:, 0]], on_edge[inv[:, 1]], on_edge[inv[:, 2]]])
    i, j = np.triu_indices(fp.shape[1], 1)
    a, b = fp[:, i].ravel(), fp[:, j].ravel()
    w = np.linalg.norm(pts[a] - pts[b], axis=1)
    ok = w > 1e-9
    g = coo_matrix((w[ok], (a[ok], b[ok])), shape=(len(pts),) * 2).tocsr()
    _, pred = dijkstra(g, directed=False, indices=src, return_predecessors=True)
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    return pts[path[::-1]]


A = np.radians(30)


def proj(p):  # 從背後、往下 30° 看
    return np.stack([-p[..., 0], p[..., 1] * np.cos(A) - p[..., 2] * np.sin(A)], axis=-1)


def depth(p):
    return -p[..., 2] * np.cos(A) + p[..., 1] * np.sin(A)


def render(ax, folder, g, s):
    d = SP / folder / g / s
    x, f = igl.read_triangle_mesh(str(d / f"{g}_{s}.obj"))
    seg = np.array((d / "segmentation.txt").read_text().splitlines())
    rec = json.loads((d / "量測.json").read_text(encoding="utf-8"))
    lm = rec["量測點"]["肩點"]
    fc = []
    for t in f:
        labs = [seg[v] for v in t if not seg[v].startswith("stitch")]
        fc.append(next((c for k, c in PAL.items() if labs and k in labs[0]), "#dddddd"))
    tri = x[f]
    order = np.argsort(depth(tri).mean(1))
    ax.add_collection(PolyCollection(proj(tri)[order], facecolors=np.asarray(fc)[order], edgecolors="#00000014", linewidths=0.2))
    path = surface_path(x, f[region(f, seg, "btorso")], lm["left"], lm["right"])
    q = proj(path)
    ax.plot(q[:, 0], q[:, 1], c="green", lw=2.4, zorder=5)
    for v in lm.values():
        p = proj(x[v])
        ax.scatter([p[0]], [p[1]], s=55, c="limegreen", ec="k", zorder=6)
    top = proj(x)[:, 1].max()
    ax.set_xlim(-40, 40)
    ax.set_ylim(top - 24, top + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return rec


fig = plt.figure(figsize=(18, 10.2))
gs = fig.add_gridspec(3, 5, height_ratios=[1.45, 0.5, 0.5], hspace=0.45, wspace=0.05)

# ── 上：誤差長條圖 ──
ax = fig.add_subplot(gs[0, :])
xi = np.arange(len(rows))
w = 0.26
for j, (key, lab, colr) in enumerate((("舊", "舊做法（全部用 Shirt）", C_OLD), ("a", "(a) 推不到才換", C_A), ("b", "(b) 落肩款全換", C_B))):
    vals = np.array([r[key]["shoulder"] - r["尺寸表"]["shoulder"] for r in rows])
    bars = ax.bar(xi + (j - 1) * w, vals, w, color=colr, label=lab, zorder=3)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + (0.12 if v >= 0 else -0.12), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, zorder=4)
ax.axhspan(-0.5, 0.5, color="#2ca02c", alpha=0.12, zorder=1)
ax.axhspan(-1, 1, color="#2ca02c", alpha=0.06, zorder=1)
ax.axhline(0, c="k", lw=0.8, zorder=2)
ax.axvline(2.5, c="#bbb", lw=1, ls="--")
ax.set_xticks(xi)
ax.set_xticklabels([f"{r['衣服']} {r['尺碼']}（表 {r['尺寸表']['shoulder']:.0f}）\n(a)：{'換落肩' if r['a換版型'] else '留 Shirt'}" for r in rows], fontsize=10.5)
ax.set_ylim(-5.8, 1.2)
ax.set_ylabel("3D 肩寬 − 尺寸表（cm）")
ax.legend(loc="lower right", fontsize=11, ncol=1)
ax.set_title("① 每個尺碼的 3D 肩寬誤差（綠底：深＝±0.5、淺＝±1 cm）", fontsize=13.5)

# ── 下：z06 五個尺碼 (a) vs (b) ──
z06 = [r for r in rows if r["衣服"] == "z06"]
for k, r in enumerate(z06):
    s = r["尺碼"]
    for row, (rule, folder, colr) in enumerate((("(a)", "_batch_drop_b12" if r["a換版型"] else "_batch", C_A),
                                                ("(b)", "_batch_drop_b12", C_B))):
        ax = fig.add_subplot(gs[1 + row, k])
        rec = render(ax, folder, "z06", s)
        kind = "落肩" if folder == "_batch_drop_b12" else "Shirt"
        diff = rec["縫好量到"]["shoulder"] - rec["尺寸表"]["shoulder"]
        ax.set_title(f"{rule} z06 {s}：{kind}", fontsize=12, color=colr, weight="bold")
        ax.text(0.5, 0.03, f"肩寬 {rec['縫好量到']['shoulder']:.1f}（表 {rec['尺寸表']['shoulder']:.0f}，{diff:+.1f}）",
                transform=ax.transAxes, ha="center", fontsize=10.5,
                bbox=dict(fc="white", ec="red" if abs(diff) > 1 else "#999", alpha=0.92))
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(colr)
            sp_.set_linewidth(2.5 if kind == "落肩" else 1)
pos = fig.axes[1].get_position()
fig.text(0.5, 0.06, "② z06 長袖 T 縫好的樣子（從背後斜上方看，綠線＝量肩寬的路徑；粗框＝落肩版型）", ha="center", fontsize=13.5)
fig.savefig(OUT, dpi=105, bbox_inches="tight")
print(OUT)
