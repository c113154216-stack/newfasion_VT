"""驗證：把縫好的 3D 衣服「拿下身體、攤平在桌上」（每條邊長不變），照尺寸表的量法量，
和 2D 版片、尺寸表、目前的 3D（穿在身上沿布面量）比。

攤平：最小化  Σ(邊長−原長)²/原長  +  w·Σz²，w 逐步加大（L-BFGS）。原長＝縫好 3D 網格的邊長，
所以布完全沒被伸縮，只是被壓平（像拿起來放到桌上）。
"""
import json
import sys
from pathlib import Path

import igl
import numpy as np
from scipy.optimize import minimize

R = Path(r"C:\Users\User\Desktop\newfashion_VT\output")
B = R / "spec_to_pattern/_batch"
OUT = Path(__file__).with_name("flat_out")
OUT.mkdir(exist_ok=True)
K_FLIP = 10.0


def flatten(x0, E, rest, F=None, sig=None, A0=None, k_flip=0.0, zmask=None):
    """k_flip>0：不准布翻面摺進去（前片朝上、後片朝下，每個三角形在桌面上的方向要對）。
    人實際平放量衣服時會把布撫平，不會讓一塊布反摺壓在底下。"""
    x = x0.copy()
    n = len(x)
    i, j = E[:, 0], E[:, 1]
    zm = np.ones(n) if zmask is None else zmask.astype(float)

    def fg(v, w):
        p = v.reshape(n, 3)
        d = p[j] - p[i]
        l = np.linalg.norm(d, axis=1) + 1e-12
        c = (l - rest) / rest
        e = np.sum(c * c * rest) + w * np.sum(zm * p[:, 2] ** 2)
        g = (2 * c / l)[:, None] * d
        G = np.zeros_like(p)
        np.add.at(G, j, g)
        np.add.at(G, i, -g)
        G[:, 2] += 2 * w * zm * p[:, 2]
        if k_flip and F is not None:
            q0, q1, q2 = p[F[:, 0], :2], p[F[:, 1], :2], p[F[:, 2], :2]
            A = 0.5 * ((q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q2[:, 0] - q0[:, 0]) * (q1[:, 1] - q0[:, 1]))
            viol = np.maximum(0.0, 0.5 * A0 - sig * A)  # 方向反了或被壓到一半以下
            e += k_flip * np.sum(viol ** 2 / A0)
            gA = -2 * k_flip * viol / A0 * sig * 0.5
            for a_, b_, c_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                pb, pc = p[F[:, b_], :2], p[F[:, c_], :2]
                gx = gA * (pb[:, 1] - pc[:, 1]); gy = gA * (pc[:, 0] - pb[:, 0])
                np.add.at(G[:, 0], F[:, a_], gx); np.add.at(G[:, 1], F[:, a_], gy)
        return e, G.ravel()

    x[:, 2] -= x[zm > 0, 2].mean()
    for w in (1e-3, 1e-2, 1e-1, 1, 10, 100):
        r = minimize(fg, x.ravel(), args=(w,), jac=True, method="L-BFGS-B",
                     options={"maxiter": 3000, "maxcor": 30})
        x = r.x.reshape(n, 3)
    return x


def face_orient(x, f, seg):
    """每個面：所屬版片、平放時該朝哪面（+1/−1，依版片在 3D 的平均法向 z；側面的版片不限制＝0）、原面積。"""
    n = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    lab = []
    for tri in f:
        labs = [seg[v] for v in tri if not seg[v].startswith("stitch")]
        lab.append(max(set(labs), key=labs.count) if labs else "")
    lab = np.array(lab)
    sig = np.zeros(len(f))
    for L in set(lab) - {""}:
        m = lab == L
        nz = n[m, 2].sum() / np.linalg.norm(n[m], axis=1).sum()
        if abs(nz) > 0.3:
            sig[m] = np.sign(nz)
    return sig, area


def labels(seg, f):
    """每個面屬於哪片版片（用不是縫線的頂點決定），每個頂點碰到哪些版片。"""
    adj = [set() for _ in seg]
    for tri in f:
        labs = [seg[v] for v in tri if not seg[v].startswith("stitch")]
        if not labs:
            continue
        lab = max(set(labs), key=labs.count)
        for v in tri:
            adj[v].add(lab)
    return adj


def corners(x, vl):
    """肩點：袖籠縫線上離領口最近的那個頂點（＝肩線、前後袖籠、袖子上緣四條縫交會的點），左右各一。
    用 GarmentCode 輸出的 vertex_labels（版片邊的 armhole / collar 標記），不靠位置猜。"""
    out = {}
    for side in ("left", "right"):
        ah = np.array(sorted(set(vl[side + "_armhole"])))
        col = np.array(sorted(set(vl[side + "_collar"])))
        d = np.linalg.norm(x[ah][:, None] - x[col][None], axis=2).min(1)
        out[side] = int(ah[d.argmin()])
    return out


def measure_flat(p, seg, adj, cn):
    """攤平後照尺寸表量（只看 xy，桌面上用尺量）。"""
    seg = np.asarray(seg)
    torso = np.array([any("torso" in s for s in a) for a in adj])
    back = np.array(["btorso" in s for s in seg])
    sleeve = np.array([any("sleeve" in s or "cuff" in s for s in a) for a in adj])
    xy = p[:, :2]
    out = {}
    L, Rr = cn["left"], cn["right"]
    out["shoulder"] = float(np.linalg.norm(xy[L] - xy[Rr]))
    # 腋下：袖子和衣身交界最低點
    pit = torso & sleeve
    armpit = xy[pit, 1].min()
    band = torso & ~sleeve & (np.abs(xy[:, 1] - (armpit - 2.5)) < 1.0)
    out["chest"] = float(np.ptp(xy[band, 0]))
    b = np.flatnonzero(back)
    hps = b[xy[b, 1].argmax()]
    out["length"] = float(xy[hps, 1] - xy[b, 1].min())
    sl = []
    for side, c in cn.items():
        s = np.flatnonzero(sleeve & np.array([any(t.startswith((side, "sl_" + side)) for t in a) for a in adj]))
        far = s[np.linalg.norm(xy[s] - xy[c], axis=1).argmax()]
        sl.append(float(np.linalg.norm(xy[far] - xy[c])))
    out["sleeve"] = float(np.mean(sl))
    return out, {"band": band, "hps": hps, "armpit": armpit}


def geo(x, f, a, b):
    f = np.asarray(f, np.int64)
    e = igl.edges(f)
    t = float(np.mean(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1))) ** 2
    return float(igl.heat_geodesic(np.asarray(x, float), f, t, np.array([a], np.int64))[b])


def run(gid, size, plot=False):
    d = B / gid / size
    m = json.loads((d / "量測.json").read_text(encoding="utf-8"))
    name = f"specb_{gid}_{size}"
    x, f = igl.read_triangle_mesh(str(d / f"{gid}_{size}.obj"))
    seg = (R / "chatgarment" / name / f"{name}_sim_segmentation.txt").read_text().split()
    assert len(seg) == len(x), (gid, size, len(seg), len(x))
    E = igl.edges(f)
    rest = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)
    keep = rest > 1e-6  # 縫線上重疊的點（長度 0）不當約束
    sig, A0 = face_orient(x, f, seg)
    use = (sig != 0) & (A0 > 1e-6)
    p = flatten(x, E[keep], rest[keep], f[use], sig[use], A0[use], k_flip=K_FLIP)
    # 攤法 B：只把衣身撫平，袖子不壓（真人平放量時袖子會自然往下斜、袖山會皺，衣身才是被撫平的部分）
    adj0 = labels(seg, f)
    tv = np.array([any("torso" in t for t in a) for a in adj0])
    ftor = np.array([all(tv[v] for v in tri) for tri in f])
    useB = use & ftor
    pB = flatten(x, E[keep], rest[keep], f[useB], sig[useB], A0[useB], k_flip=K_FLIP, zmask=tv)
    l2 = np.linalg.norm(p[E[keep, 0]] - p[E[keep, 1]], axis=1)
    strain = l2 / rest[keep]
    adj = labels(seg, f)
    import yaml
    vl = yaml.safe_load(open(R / "chatgarment" / name / f"{name}_vertex_labels.yaml"))
    cn = corners(x, vl)
    # 3D 穿著時布有沒有被撐開：不碰縫線的邊，縫好長度 ÷ 版片（boxmesh 平放）長度
    bx, _ = igl.read_triangle_mesh(str(R / "chatgarment" / name / f"{name}_boxmesh.obj"))
    st = np.array([s.startswith("stitch") for s in seg])
    inner = ~(st[E[:, 0]] | st[E[:, 1]])
    lb = np.linalg.norm(bx[E[inner, 0]] - bx[E[inner, 1]], axis=1)
    worn = rest[inner] / lb
    flat, aux = measure_flat(p, seg, adj, cn)
    flatB, _ = measure_flat(pB, seg, adj, cn)
    res = {
        "衣服": m["衣服"], "尺碼": size,
        "尺寸表": m["尺寸表"], "版片量到": m["版片量到"], "縫好量到(穿著沿布面)": m["縫好量到"],
        "攤平量到": {k: round(v, 1) for k, v in flat.items()},
        "只撫平衣身量到": {k: round(v, 1) for k, v in flatB.items() if k != "sleeve"},
        "攤平品質": {"翻面的三角形": int(np.sum((sig != 0) & (sig * (0.5 * ((p[f[:, 1], 0] - p[f[:, 0], 0]) * (p[f[:, 2], 1] - p[f[:, 0], 1]) - (p[f[:, 2], 0] - p[f[:, 0], 0]) * (p[f[:, 1], 1] - p[f[:, 0], 1]))) < 0))),"z 厚度": round(float(np.ptp(p[:, 2])), 2),
                  "邊長比 1%/50%/99%": [round(float(q), 4) for q in np.percentile(strain, [1, 50, 99])]},
        "穿著時布的伸縮(版片=1) 5%/50%/95%": [round(float(q), 3) for q in np.percentile(worn, [5, 50, 95])],
        "穿著時布平均面積比": round(float(np.mean(worn ** 2)), 3),
        "肩點沿布面(穿著)": round(geo(x, f, cn["left"], cn["right"]), 1),
        "肩點直線(穿著)": round(float(np.linalg.norm(x[cn["left"]] - x[cn["right"]])), 1),
    }
    np.save(OUT / f"{gid}_{size}_flat.npy", p)
    if plot:
        draw(p, f, seg, cn, aux, res, OUT / f"{gid}_{size}_攤平.png")
    return res


def draw(p, f, seg, cn, aux, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    fig, ax = plt.subplots(figsize=(10, 7))
    col = {"btorso": "#9ecae1", "ftorso": "#c6dbef", "sleeve": "#fdd0a2", "cuff": "#fdae6b"}
    fc = []
    for tri in f:
        labs = [seg[v] for v in tri if not seg[v].startswith("stitch")]
        lab = labs[0] if labs else "stitch"
        fc.append(next((c for k, c in col.items() if k in lab), "#dddddd"))
    from matplotlib.collections import PolyCollection
    # 前片在上層（從正面看平放的衣服），後片在下
    order = np.argsort([0 if "b" in c else 1 for c in fc])
    ax.add_collection(PolyCollection(p[f[order]][:, :, :2], facecolors=np.array(fc)[order],
                                     edgecolors="#00000018", linewidths=0.3))
    a, b = p[cn["left"], :2], p[cn["right"], :2]
    s, t = res["攤平量到"], res["版片量到"]
    ax.annotate("", a, b, arrowprops=dict(arrowstyle="<->", color="crimson", lw=2))
    ax.text(*(a + b) / 2 + [0, 2], f"肩寬 {s['shoulder']}（版片 {t['shoulder']}）", color="crimson", ha="center")
    band = aux["band"]
    y0 = p[band, 1].mean()
    x0, x1 = p[band, 0].min(), p[band, 0].max()
    ax.annotate("", (x0, y0), (x1, y0), arrowprops=dict(arrowstyle="<->", color="navy", lw=2))
    ax.text((x0 + x1) / 2, y0 - 4, f"胸寬 {s['chest']}（版片 {t['chest']}）", color="navy", ha="center")
    h = aux["hps"]
    yb = p[:, 1].min()
    ax.annotate("", (p[h, 0], p[h, 1]), (p[h, 0], p[h, 1] - s["length"]),
                arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2))
    ax.text(p[h, 0] + 1.5, p[h, 1] - s["length"] * 0.6, f"衣長 {s['length']}\n（版片 {t['length']}）", color="darkgreen")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(f"{res['衣服']} {res['尺碼']}：縫好的 3D 衣服攤平（邊長不變），單位 cm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    items = sys.argv[1:] or [f"{g.name}/{s.name}" for g in sorted(B.glob("z0*")) for s in sorted(g.iterdir()) if s.is_dir()]
    allres = []
    for it in items:
        gid, size = it.split("/")
        try:
            r = run(gid, size, plot=True)
        except Exception as ex:
            print(f"{gid} {size} 跳過：{ex!r}", flush=True)
            continue
        (OUT / f"{gid}_{size}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        allres.append(r)
        k = ("shoulder", "chest", "length", "sleeve")
        print(f"{gid} {size:4s}", " | ".join(
            f"{kk} 表{r['尺寸表'].get(kk, '-')} 版{round(r['版片量到'][kk],1)} 平{r['攤平量到'][kk]} 穿{round(r['縫好量到(穿著沿布面)'][kk],1)}" for kk in k),
            "| 衣身撫平 肩", r["只撫平衣身量到"]["shoulder"], "胸", r["只撫平衣身量到"]["chest"], "長", r["只撫平衣身量到"]["length"], "| 攤平邊長", r["攤平品質"]["邊長比 1%/50%/99%"], "| 穿著伸縮", r["穿著時布的伸縮(版片=1) 5%/50%/95%"], "| 真肩點布面", r["肩點沿布面(穿著)"], flush=True)
    if not sys.argv[1:]:
        (OUT / "攤平總表.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
