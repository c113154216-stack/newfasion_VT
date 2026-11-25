# -*- coding: utf-8 -*-
"""周誌用的肩寬圖（2026-11-25）：z06 M 的真實縫合結果，三個量不準的問題各一張、加標註。
  周誌11_肩點.png      問題 1：舊肩點（後片最外側）vs 真肩點（後片袖籠縫線最上端）
  周誌11_距離.png      問題 2：沿布面的最短路徑，精確算法 vs heat geodesic
  周誌11_攤平.png      問題 3：縫好 → 攤平 → 再縫兩次，肩角的布被壓多少
資料：output/spec_to_pattern/_batch、_batch_before_resew，縫合資料夾用 11-25 的備份
（output/chatgarment/specb_z06_M 已經被落肩版型的實驗蓋掉）。
跑法（newfashion_VT 根目錄）：.venv\\Scripts\\python docs\\驗證_尺寸\\fig_shoulder_journal.py
"""
import json
import pickle
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flows" / "size_chart"))
sys.path.insert(0, str(ROOT / "flows" / "body_fit"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

from sew import mesh_edges, rest_lengths  # noqa: E402
from spec_measure import STEINER_K, _pairs, region, shoulder, sleeve, surface_dist  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
OUTS = ROOT / "output" / "spec_to_pattern"
OUT_DIR = ROOT / "docs" / "images" / "size_chart"
g, s = "z06", "M"
name = f"specb_{g}_{s}"
D = ROOT / "output" / "chatgarment" / "_backup_1125_before_drop" / name
rec = json.loads((OUTS / "_batch" / g / s / "量測.json").read_text(encoding="utf-8"))
lm = rec["量測點"]
pat = rec["版片量到"]
box, f = igl.read_triangle_mesh(str(D / f"{name}_boxmesh.obj"))
seg = np.array((OUTS / "_batch" / g / s / "segmentation.txt").read_text().splitlines())
E = mesh_edges(f)
rest = rest_lengths(box, E, pickle.loads((D / f"{name}_orig_lens.pickle").read_bytes()))


def load(folder):
    x, _ = igl.read_triangle_mesh(str(OUTS / folder / g / s / f"{g}_{s}.obj"))
    assert len(x) == len(box)
    return x


x_final = load("_batch")
x_unfold = load("_batch_before_resew")
x_sewn, _ = igl.read_triangle_mesh(str(D / f"{name}_sewn.obj"))

pal = {"btorso": "#9ecae1", "ftorso": "#cbbbe6", "sleeve": "#fdd0a2", "cuff": "#fdae6b", "collar": "#c7e9c0"}
face_col = []
for t in f:
    labs = [seg[v] for v in t if not seg[v].startswith("stitch")]
    face_col.append(next((c for k, c in pal.items() if labs and k in labs[0]), "#dddddd"))
face_col = np.array([colors.to_rgba(c) for c in face_col])


def make_proj(deg, yaw=0.0):
    a, y = np.radians(deg), np.radians(yaw)

    def rot(p):  # 繞身體直立軸轉 yaw 度（看側後方）
        p = np.asarray(p, float)
        return np.stack([p[..., 0] * np.cos(y) + p[..., 2] * np.sin(y), p[..., 1],
                         -p[..., 0] * np.sin(y) + p[..., 2] * np.cos(y)], axis=-1)

    def proj(p):  # 從背後看，往下 deg 度
        p = rot(p)
        return np.stack([-p[..., 0], p[..., 1] * np.cos(a) - p[..., 2] * np.sin(a)], axis=-1)

    def depth(p):
        p = rot(p)
        return -p[..., 2] * np.cos(a) + p[..., 1] * np.sin(a)
    return proj, depth


def draw(ax, x, fc, proj, depth, center=None, half=None, edge="#00000020", fsel=None):
    fsel = np.ones(len(f), bool) if fsel is None else fsel
    tri = x[f[fsel]]
    order = np.argsort(depth(tri).mean(1))
    ax.add_collection(PolyCollection(proj(tri)[order], facecolors=np.asarray(fc)[fsel][order], edgecolors=edge, linewidths=0.25))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if center is None:
        q = proj(x)
        ax.set_xlim(q[:, 0].min() - 3, q[:, 0].max() + 3)
        ax.set_ylim(q[:, 1].min() - 3, q[:, 1].max() + 3)
    else:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half * 0.75, center[1] + half * 0.75)


def mark(ax, proj, p, colr, mk, lab, size=150):
    q = proj(p)
    ax.scatter([q[0]], [q[1]], s=size, c=colr, marker=mk, zorder=8, edgecolors="k", linewidths=0.8, label=lab)
    return q


def note(ax, xy, text, xytext, colr="k"):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", fontsize=10.5, color=colr, zorder=9,
                arrowprops=dict(arrowstyle="->", color=colr, lw=1.3),
                bbox=dict(fc="white", ec=colr, alpha=0.92, boxstyle="round,pad=0.3"))


def steiner_path(x, ff, src, dst, k=STEINER_K):
    """spec_measure.surface_dist 同一個算法，多回傳路徑上的點（畫圖用）。"""
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
    gph = coo_matrix((w[ok], (a[ok], b[ok])), shape=(len(pts),) * 2).tocsr()
    d, pred = dijkstra(gph, directed=False, indices=src, return_predecessors=True)
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    return float(d[dst]), pts[path[::-1]]


def vertex_path(x, ff, src, dst):
    """只走網格的邊（不加點）的最短路。"""
    e = np.sort(np.vstack([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    w = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
    gph = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(x),) * 2).tocsr()
    d, pred = dijkstra(gph, directed=False, indices=src, return_predecessors=True)
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    return float(d[dst]), x[path[::-1]]


# ───────────── 問題 1：肩點 ─────────────
x = x_final
torso = np.char.find(seg, "torso") >= 0
back = np.char.find(seg, "btorso") >= 0
slv = (np.char.find(seg, "sleeve") >= 0) | (np.char.find(seg, "cuff") >= 0)
near = slv & (np.abs(x[:, 0]) < np.abs(x[torso, 0]).max() + 4)
armpit_y = x[near, 1].min()
cand = np.flatnonzero(back & (x[:, 1] > armpit_y + 3) & (x[:, 0] < 0))
old_r = cand[np.abs(x[cand, 0]).argmax()]
cand = np.flatnonzero(back & (x[:, 1] > armpit_y + 3) & (x[:, 0] > 0))
old_l = cand[np.abs(x[cand, 0]).argmax()]
new_r, new_l = lm["肩點"]["right"], lm["肩點"]["left"]
drop = float(np.linalg.norm(x[old_r] - x[new_r]))

mask = region(f, seg, "btorso")
w_new = shoulder(x, f, seg, lm)
w_old = surface_dist(x, f, old_l, old_r, mask)
d_r, d_l = x[old_r] - x[new_r], x[old_l] - x[new_l]
stitches = json.loads((D / f"{name}_specification.json").read_text(encoding="utf-8"))["pattern"]["stitches"]
tags = [set(t.split(",")) for t in seg]


def seam(side, a, b, upper=None):
    """兩片版片之間那條縫線上的頂點；upper=True 只取上面那條（肩線、袖子上緣），False 取下面那條。"""
    ids = {f"stitch_{i}" for i, p, q in _pairs(stitches) if {p, q} == {f"{side}_{a}", f"{side}_{b}"}}
    v = np.array([k for k, t in enumerate(tags) if t & ids], int)
    if upper is not None and len(v):
        mid = x[v, 1].mean()
        v = v[x[v, 1] > mid] if upper else v[x[v, 1] <= mid]
    return v


SEAMS = [("肩線（前片＋後片）", ("ftorso", "btorso", True), "black"),
         ("後袖籠（後片＋袖子）", ("btorso", "sleeve_b", None), "purple"),
         ("袖子上緣（袖子前＋後）", ("sleeve_f", "sleeve_b", True), "saddlebrown")]
PART_NAME = {"btorso": "後片", "ftorso": "前片", "sleeve": "袖子"}

proj, depth = make_proj(35)
fig = plt.figure(figsize=(17, 13.5))
gsp = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.25])
ax = fig.add_subplot(gsp[0, :])
draw(ax, x, face_col, proj, depth)
_, po = steiner_path(x, f[mask], old_l, old_r)
_, pn = steiner_path(x, f[mask], new_l, new_r)
q = proj(pn)
ax.plot(q[:, 0], q[:, 1], c="green", lw=3, zorder=7, label=f"真肩點量：{w_new:.1f} cm")
q = proj(po)
ax.plot(q[:, 0], q[:, 1], c="red", lw=2, ls="--", zorder=7, label=f"舊肩點量：{w_old:.1f} cm（少 {w_new - w_old:.1f}）")
for v in (old_l, old_r):
    mark(ax, proj, x[v], "red", "X", None)
for v in (new_l, new_r):
    mark(ax, proj, x[v], "limegreen", "o", None)
qq = proj(x)
ax.set_ylim(qq[:, 1].max() - 42, qq[:, 1].max() + 4)
ax.legend(loc="lower center", fontsize=11.5, framealpha=0.95, title=f"版片上量到 {pat['shoulder']} cm", title_fontsize=11.5)
ax.set_title("① 整件（z06 長袖 T，M），從背後斜上方看：肩寬＝左肩點 → 右肩點，沿後片布面量", fontsize=13.5)
ax.text(0.01, 0.98, "藍：後片　紫：前片　橘：袖子", transform=ax.transAxes, va="top", fontsize=11, bbox=dict(fc="white", ec="#999"))
for k, (side, sd, vo, vn, dd) in enumerate((("右肩", "right", old_r, new_r, d_r), ("左肩", "left", old_l, new_l, d_l))):
    pr, de = make_proj(90)  # 從正上方往下看：畫面上方＝背後，下方＝胸前
    ytop = x[vn, 1] - 10
    fsel = np.all(x[f][:, :, 1] > ytop, axis=1)  # 只畫肩膀這一段，下面的衣身會從上面看穿
    ax = fig.add_subplot(gsp[1, k])
    c = pr(x[vn]) + [(-1 if k == 0 else 1) * 2.0, 0.0]
    half = 14
    draw(ax, x, face_col, pr, de, center=c, half=half, fsel=fsel)
    for lab, (a_, b_, up), colr in SEAMS:
        v = seam(sd, a_, b_, up)
        v = v[x[v, 1] > ytop]
        qv = pr(x[v])
        ax.scatter(qv[:, 0], qv[:, 1], s=18, c=colr, zorder=6, label=lab)
    qa = pr(x)
    inwin = (np.abs(qa[:, 0] - c[0]) < half * 0.8) & (np.abs(qa[:, 1] - c[1]) < half * 0.6) & (x[:, 1] > ytop + 3)
    for key, nm in PART_NAME.items():
        sel = inwin & np.array([t.startswith(f"{sd}_{key}") for t in seg])
        if sel.sum() > 10:
            m = np.median(qa[sel], axis=0)
            ax.text(m[0], m[1], nm, fontsize=15, weight="bold", ha="center", va="center", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75))
    qn = mark(ax, pr, x[vn], "limegreen", "o", None, 260)
    qo = mark(ax, pr, x[vo], "red", "X", None, 280)
    ax.annotate("", xy=qo, xytext=qn, arrowprops=dict(arrowstyle="->", color="red", lw=2, shrinkA=9, shrinkB=9), zorder=7)
    ax.set_title(f"{'②' if k == 0 else '③'} 放大{side}（從正上方往下看）", fontsize=13.5)
    ax.text(0.99 if k == 0 else 0.01, 0.97, "↑ 背後\n↓ 胸前", transform=ax.transAxes, ha="right" if k == 0 else "left",
            va="top", fontsize=12, bbox=dict(fc="white", ec="#999"))
    txt = f"舊肩點離真肩點：往後 {-dd[2]:.1f}、往內 {abs(dd[0]):.1f}" + (f"、往下 {-dd[1]:.1f}" if -dd[1] > 0.5 else "") + " cm"
    ax.text(0.5, 0.02, txt, transform=ax.transAxes, ha="center", fontsize=11.5, va="bottom",
            bbox=dict(fc="white", ec="red", alpha=0.95))
    ax.legend(loc="upper left" if k == 0 else "upper right", fontsize=10.5, framealpha=0.95, markerscale=2)
fig.legend(handles=[plt.Line2D([], [], marker="X", ls="", ms=12, mfc="red", mec="k", label="舊肩點：後片在腋下以上最外側的點"),
                    plt.Line2D([], [], marker="o", ls="", ms=12, mfc="limegreen", mec="k",
                               label="真肩點：肩線、後袖籠、袖子上緣三條縫線交會的點")],
           loc="lower center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT_DIR / "周誌11_肩點.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("問題1 新", round(w_new, 2), "舊", round(w_old, 2), "右", np.round(d_r, 1), "左", np.round(d_l, 1))


# ───────────── 問題 2：距離算法 ─────────────
mask = region(f, seg, "btorso")
fb = f[mask]
exact, path = steiner_path(x, fb, new_l, new_r)
edge_only, epath = vertex_path(x, fb, new_l, new_r)
used = np.unique(fb)
remap = -np.ones(len(x), int)
remap[used] = np.arange(len(used))
xs, fs = x[used], remap[fb]
el = np.linalg.norm(xs[fs[:, 0]] - xs[fs[:, 1]], axis=1).mean()
heat = igl.heat_geodesic(xs, fs, el ** 2, np.array([remap[new_l]], dtype=fs.dtype))
heat_d = float(heat[remap[new_r]])
print("問題2 精確", round(exact, 2), "只走邊", round(edge_only, 2), "heat", round(heat_d, 2), "版片", pat["shoulder"])

proj, depth = make_proj(30)
fig = plt.figure(figsize=(16, 7.2))
gsp = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
ax = fig.add_subplot(gsp[0])
fc = face_col.copy()
fc[~mask] = colors.to_rgba("#eeeeee")
draw(ax, x, fc, proj, depth)
q = proj(path)
ax.plot(q[:, 0], q[:, 1], c="green", lw=3, zorder=7, label=f"精確最短路（現在）：{exact:.1f} cm")
q = proj(epath)
ax.plot(q[:, 0], q[:, 1], c="purple", lw=1.6, ls="--", zorder=7, label=f"只沿網格的邊走：{edge_only:.1f} cm（繞遠路，偏長）")
for v in (new_l, new_r):
    mark(ax, proj, x[v], "limegreen", "o", None)
qq = proj(x)
ax.set_ylim(qq[:, 1].max() - 42, qq[:, 1].max() + 4)
ax.plot([], [], c="none", label=f"libigl heat geodesic（舊，只有數字沒有路徑）：{heat_d:.1f} cm（短 {exact - heat_d:.1f}）")
ax.legend(loc="lower center", fontsize=11, framealpha=0.95, title=f"版片上量到 {pat['shoulder']} cm", title_fontsize=11)
ax.set_title("肩寬：左肩點 → 右肩點，只沿後片（藍）的布面量", fontsize=13)

ax = fig.add_subplot(gsp[1])
tri = np.array([[0, 0], [10, 0], [4, 8]], float)
ax.add_patch(plt.Polygon(tri, fc="#9ecae1", ec="k", lw=1.5))
k = STEINER_K
tt = np.arange(1, k + 1) / (k + 1)
ep = [tri[a] * (1 - tt)[:, None] + tri[b] * tt[:, None] for a, b in ((0, 1), (1, 2), (2, 0))]
allp = np.vstack([tri] + ep)
for i in range(len(allp)):
    for j in range(i + 1, len(allp)):
        ax.plot(*allp[[i, j]].T, c="#999999", lw=0.35, zorder=1)
ax.scatter(*tri.T, s=60, c="k", zorder=3, label="原本的頂點")
ax.scatter(*np.vstack(ep).T, s=30, c="orange", zorder=3, label=f"每條邊多加 {k} 個點")
a_, b_ = ep[2][2], ep[0][3]  # 左邊上一個點 → 底邊上一個點：直接斜穿
ax.plot(*np.array([a_, b_]).T, c="green", lw=3, zorder=4, label="可以斜穿三角形")
ax.plot(*np.array([a_, tri[0], b_]).T, c="purple", lw=2, ls="--", zorder=4, label="只走邊：要繞到頂點")
ax.set_aspect("equal")
ax.set_xlim(-1, 11)
ax.set_ylim(-4.2, 8.5)
ax.axis("off")
ax.legend(loc="lower center", fontsize=10.5, ncol=2, bbox_to_anchor=(0.5, -0.02))
ax.set_title("精確算法：每條邊加點、三角形內兩兩相連\n再用 Dijkstra 找最短路（平的版片上誤差 < 0.5%）", fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "周誌11_距離.png", dpi=110, bbox_inches="tight")
plt.close(fig)


# ───────────── 問題 3：攤平把肩角縮掉 ─────────────
def edge_ratio_faces(x):
    r = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1) / rest
    emap = {(a, b): v for (a, b), v in zip(map(tuple, E), r)}
    return np.array([np.mean([emap[tuple(sorted((t[i], t[(i + 1) % 3])))] for i in range(3)]) for t in f]), r


def ring_min(r, v, x, radius=2.5):
    """肩點半徑 radius cm 內的邊，最短的被壓到原長幾成。"""
    near_e = np.linalg.norm((x[E[:, 0]] + x[E[:, 1]]) / 2 - x[v], axis=1) < radius
    return float(np.median(r[near_e]))


stages = [(x_sewn, "① 縫好、還沒攤平"), (x_unfold, "② 攤平尖角（原本的做法）"), (x_final, "③ 攤平後再縫兩次（現在）")]
norm = colors.TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.15)
proj, depth = make_proj(40)
fig, axs = plt.subplots(1, 3, figsize=(18, 6.6))
for ax, (xx, title) in zip(axs, stages):
    fr, r = edge_ratio_faces(xx)
    cc = proj(xx[new_r])
    draw(ax, xx, cm.RdBu(norm(fr)), proj, depth, center=cc, half=11)
    mark(ax, proj, xx[new_r], "limegreen", "o", None, 180)
    sh, sl = shoulder(xx, f, seg, lm), sleeve(xx, f, seg, lm)
    ax.set_title(title, fontsize=13)
    ax.text(0.02, 0.03, f"肩寬 {sh:.1f}（版片 {pat['shoulder']}）\n袖長 {sl:.1f}（版片 {pat['sleeve']}）\n"
            f"肩點附近的邊：原長的 {ring_min(r, new_r, xx):.2f}",
            transform=ax.transAxes, fontsize=11, va="bottom", bbox=dict(fc="white", ec="#999", alpha=0.93))
    print("問題3", title, round(sh, 2), round(sl, 2), round(ring_min(r, new_r, xx), 3))
note(axs[0], proj(x_sewn[new_r]), "四條縫線交會，\n縫好時翻折成尖角", (proj(x_sewn[new_r])[0] + 2, proj(x_sewn[new_r])[1] + 6), "k")
note(axs[1], proj(x_unfold[new_r]), "攤平＝把尖角的點往鄰居平均拉，\n附近的布被縮短（紅）", (proj(x_unfold[new_r])[0] - 10, proj(x_unfold[new_r])[1] + 6.5), "darkred")
note(axs[2], proj(x_final[new_r]), "從攤平的位置再縫，\n布回到原長，變成平順的鼓起", (proj(x_final[new_r])[0] - 10, proj(x_final[new_r])[1] + 6.5), "navy")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.RdBu), ax=axs, fraction=0.025, pad=0.01,
             label="3D 邊長 ÷ 版片原長（紅＝被壓短，藍＝被拉長）")
fig.suptitle("z06 長袖 T（M）右肩放大，從背後斜上方看", fontsize=13, y=0.99)
fig.savefig(OUT_DIR / "周誌11_攤平.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("輸出到", OUT_DIR)
