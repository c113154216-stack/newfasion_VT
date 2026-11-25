"""圖解：3D 肩寬為什麼量不準、怎麼修（2026-11-24）。
① 肩點抓在哪  ② 攤平尖角把肩角縮掉、再縫兩次救回來  ③ 四個版本 3D 肩寬 − 版片  ④ 2D 版片推不到
跑法（newfashion_VT 根目錄，先跑過整批和各版本備份）：
    .venv\\Scripts\\python docs\\驗證_尺寸\\fig_shoulder.py
"""
import glob
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
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from sew import mesh_edges, rest_lengths  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
OUTS = ROOT / "output" / "spec_to_pattern"
R = ROOT / "output" / "chatgarment"
g, s = "z06", "M"
name = f"specb_{g}_{s}"
d = R / name
rec = json.loads((OUTS / "_batch" / g / s / "量測.json").read_text(encoding="utf-8"))
lm = rec["量測點"]
box, f = igl.read_triangle_mesh(str(d / f"{name}_boxmesh.obj"))
seg = np.array((d / f"{name}_sim_segmentation.txt").read_text().splitlines())
E = mesh_edges(f)
rest = rest_lengths(box, E, pickle.loads((d / f"{name}_orig_lens.pickle").read_bytes()))
vl = yaml.safe_load(open(d / f"{name}_vertex_labels.yaml"))


def mesh(folder):
    x, _ = igl.read_triangle_mesh(str(OUTS / folder / g / s / f"{g}_{s}.obj"))
    assert len(x) == len(box)
    return x


x_final = mesh("_batch")
x_unfold = mesh("_batch_before_resew")  # 縫好 → 攤平（原本做法）
x_sewn = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # 縫好、還沒攤平（可選，實驗存的 npy）
x_sewn = np.load(x_sewn) if x_sewn else None

A = np.radians(40)  # 從背後、往下 40° 看


def proj(p):
    return np.stack([-p[..., 0], p[..., 1] * np.cos(A) - p[..., 2] * np.sin(A)], axis=-1)


def view(ax, x, facecol, title, center, half):
    depth = (-x[f][:, :, 2] * np.cos(A) + x[f][:, :, 1] * np.sin(A)).mean(1)
    order = np.argsort(depth)
    ax.add_collection(PolyCollection(proj(x[f])[order], facecolors=np.asarray(facecol)[order],
                                     edgecolors="#00000018", linewidths=0.25))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half * 0.8, center[1] + half * 0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def face_ratio(x):
    r = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1) / rest
    emap = {(a, b): v for (a, b), v in zip(E, r)}
    return np.array([np.mean([emap[tuple(sorted((t[i], t[(i + 1) % 3])))] for i in range(3)]) for t in f])


pal = {"btorso": "#9ecae1", "ftorso": "#deebf7", "sleeve": "#fdd0a2", "cuff": "#fdae6b"}
fc = []
for t in f:
    labs = [seg[v] for v in t if not seg[v].startswith("stitch")]
    fc.append(next((c for k, c in pal.items() if labs and k in labs[0]), "#cccccc"))

fig = plt.figure(figsize=(19, 12))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])

# ① 肩點抓在哪
torso = np.array(["torso" in t for t in seg])
back = np.array(["btorso" in t for t in seg])
sleeve = np.array(["sleeve" in t or "cuff" in t for t in seg])
x = x_final
near = sleeve & (np.abs(x[:, 0]) < np.abs(x[torso, 0]).max() + 4)
armpit = x[near, 1].min()
t_ = np.flatnonzero(back & (x[:, 1] > armpit + 3) & (x[:, 0] < 0))
old_tip = t_[np.abs(x[t_, 0]).argmax()]
ah = np.array(sorted(set(vl["right_armhole"])))
col = np.array(sorted(set(vl["right_collar"])))
old_lab = ah[np.linalg.norm(x[ah][:, None] - x[col][None], axis=2).min(1).argmin()]
c = lm["肩點"]["right"]
C = proj(x[c])
ax = fig.add_subplot(gs[0, 0])
view(ax, x, fc, "① 肩點抓在哪（z06 M 右肩，背後斜上方）", C, 13)
for p, colr, mk, lab in ((old_tip, "red", "X", "舊：後片最外側的點"), (old_lab, "orange", "s", "11/23：袖籠上離領口最近"),
                         (c, "green", "o", "新：後片袖籠縫線最上端")):
    q = proj(x[p])
    ax.scatter([q[0]], [q[1]], s=130, c=colr, marker=mk, zorder=5, edgecolors="k", linewidths=0.7, label=lab)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

# ② 三個階段的肩角
norm = colors.TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.15)
stages = [(x_unfold, "② 攤平尖角後（原本做法）"), (x_final, "③ 攤平 → 再縫兩次（現在）")]
if x_sewn is not None:
    stages.insert(0, (x_sewn, "縫好、還沒攤平"))
for k, (xx, title) in enumerate(stages[-2:]):
    ax = fig.add_subplot(gs[0, 1 + k])
    cc = proj(xx[c])
    view(ax, xx, cm.RdBu(norm(face_ratio(xx))), title, cc, 13)
    ax.scatter([cc[0]], [cc[1]], s=110, c="green", edgecolors="k", zorder=6)
    from spec_measure import shoulder, sleeve as sleeve_len
    sh, sl = shoulder(xx, f, seg, lm), sleeve_len(xx, f, seg, lm)
    ax.text(0.02, 0.03, f"肩寬 {sh:.1f}（版片 {rec['版片量到']['shoulder']}）\n袖長 {sl:.1f}（版片 {rec['版片量到']['sleeve']}）",
            transform=ax.transAxes, fontsize=10, va="bottom", bbox=dict(fc="white", ec="#999", alpha=0.9))
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.RdBu), ax=fig.axes[-1], fraction=0.05, label="3D 邊長 ÷ 版片原長（紅＝被壓縮）")

ax = fig.add_subplot(gs[0, 3])
ax.axis("off")
ax.text(0, 1, "為什麼會縮、怎麼修", fontsize=13, weight="bold", va="top")
ax.text(0, 0.9, "• 縫好的時候肩寬 62.0、袖長 53.9，跟版片一樣\n"
                "• 肩點是四條縫線交會處，縫好時那裡會翻折成尖角\n"
                "• 攤平尖角＝把附近的點往鄰居平均拉，\n   肩角附近的布被縮到原長的 0.74\n"
                "   → 肩寬少 4.0、袖長少 2.7\n"
                "• 修法：攤平後從攤平的位置再縫兩次\n   （彎曲剛度 3），布回到原長、\n   形成平順的鼓起，不再折成尖角\n"
                "• 試過沒用的：拿掉彎曲、壓短的邊變硬、\n   拿掉身體、只沿法向攤平、攤平後拉回邊長",
        fontsize=10.5, va="top", linespacing=1.6)

# ③ 四個版本
order = ["z01", "z02", "z04", "z05", "z06", "z07"]
sizes = ["XXS", "XS", "S", "M", "L", "XL", "2XL"]


def load(folder):
    out = {}
    for p in glob.glob(str(OUTS / folder / "z0*" / "*" / "量測.json")):
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        if "shoulder" in r["尺寸表"]:
            out[(r["衣服"][:3], r["尺碼"])] = r
    return out


vers = [("_batch_old_1123", "11/23 舊版（修正迴圈＋舊量法）", "#d62728"), ("_batch_before_resew", "新量法、只攤平", "#ff9f40"),
        ("_batch_resew1", "再縫 1 次", "#9ecae1"), ("_batch", "再縫 2 次（現在）", "#2ca02c")]
data = [load(v[0]) for v in vers]
keys = sorted(data[-1], key=lambda k: (order.index(k[0]), sizes.index(k[1])))
ax = fig.add_subplot(gs[1, 0:3])
xi = np.arange(len(keys))
w = 0.2
for j, ((_, lab, colr), D) in enumerate(zip(vers, data)):
    vals = [D[k]["縫好量到"]["shoulder"] - D[k]["尺寸表"]["shoulder"] if k in D else np.nan for k in keys]
    ax.bar(xi + (j - 1.5) * w, vals, w, color=colr, label=lab)
cap = [data[-1][k]["版片量到"]["shoulder"] - data[-1][k]["尺寸表"]["shoulder"] for k in keys]
ax.scatter(xi + 1.5 * w, cap, marker="_", s=180, c="k", zorder=5, label="2D 版片 − 尺寸表（推不到的部分）")
ax.axhline(0, c="k", lw=0.8)
ax.axhspan(-1, 1, color="#2ca02c", alpha=0.07)
ax.set_ylim(-9, 1.5)
ax.set_xticks(xi)
ax.set_xticklabels([f"{a}\n{b}" for a, b in keys], fontsize=8.5)
ax.set_ylabel("3D 肩寬 − 尺寸表（cm）")
ax.legend(fontsize=9, loc="lower left", ncol=2)
fin = np.array([data[-1][k]["縫好量到"]["shoulder"] - data[-1][k]["版片量到"]["shoulder"] for k in keys])
ax.set_title(f"④ 每個尺碼的 3D 肩寬誤差（綠底＝±1 cm）。現在：3D 對版片 {np.sum(abs(fin) <= 1)}/{len(fin)} 在 1 cm 內，"
             f"對尺寸表 {np.sum(abs(fin + np.array(cap)) <= 1)}/{len(fin)}", fontsize=11.5)
# 舊版超出圖的標數字
for i, k in enumerate(keys):
    v = data[0][k]["縫好量到"]["shoulder"] - data[0][k]["尺寸表"]["shoulder"] if k in data[0] else 0
    if v < -9:
        ax.text(i - 1.5 * w, -8.8, f"{v:.0f}", ha="center", fontsize=7.5, color="#d62728")

# ④ 2D 推不到
ax = fig.add_subplot(gs[1, 3])
capk = [k for k in keys if k[0] in ("z04", "z06")]
old2d = {k: data[1].get(k) for k in capk}
yi = np.arange(len(capk))
before = []
for k in capk:  # 側縫沒移之前（11-23 舊版是修正迴圈推過的，不能用）：用 11-24 實測值
    before.append({"z06 XXS": 53.2, "z06 XS": 55.2, "z06 S": 57.3, "z06 M": 58.9, "z06 L": 61.1,
                   "z04 S": 52.6, "z04 M": 54.6, "z04 L": 56.5}[f"{k[0]} {k[1]}"])
ax.barh(yi + 0.2, before, 0.38, color="#bbbbbb", label="側縫偏後（GarmentCode 原本）")
ax.barh(yi - 0.2, [data[-1][k]["版片量到"]["shoulder"] for k in capk], 0.38, color="#4a90d9", label="側縫移到正側邊（現在）")
ax.scatter([data[-1][k]["尺寸表"]["shoulder"] for k in capk], yi, marker="|", s=400, c="crimson", label="尺寸表", zorder=5)
ax.set_yticks(yi)
ax.set_yticklabels([f"{a} {b}" for a, b in capk])
ax.set_xlim(48, 66)
ax.invert_yaxis()
ax.set_xlabel("2D 版片肩寬（cm）")
ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
ax.set_title("⑤ 2D 推不到：z04 肩寬比胸寬大，做不到", fontsize=11.5)
fig.tight_layout()
out = Path(__file__).with_name("肩寬_圖解.png")
fig.savefig(out, dpi=105)
print(out)
