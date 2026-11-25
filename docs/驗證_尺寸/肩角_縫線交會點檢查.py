"""肩點附近每條邊：兩端屬於哪片、原長（orig_lens 或 boxmesh）、boxmesh 長度、縫好 3D 長度。"""
import sys, json, pickle
import numpy as np, igl
from pathlib import Path
sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\body_fit")
from sew import mesh_edges
R = Path(r"C:\Users\User\Desktop\newfashion_VT\output\chatgarment")
B = Path(r"C:\Users\User\Desktop\newfashion_VT\output\spec_to_pattern\_batch")
g, s = sys.argv[1], sys.argv[2]
name = f"specb_{g}_{s}"; d = R / name
rec = json.loads((B / g / s / "量測.json").read_text(encoding="utf-8"))
v, f = igl.read_triangle_mesh(str(d / f"{name}_boxmesh.obj"))
x, _ = igl.read_triangle_mesh(str(B / g / s / f"{g}_{s}.obj"))
seg = (d / f"{name}_sim_segmentation.txt").read_text().splitlines()
ol = pickle.loads((d / f"{name}_orig_lens.pickle").read_bytes())
E = mesh_edges(f)
c = rec["量測點"]["肩點"]["right"]
ring = set(np.unique(f[np.any(f == c, axis=1)]))
ring2 = set(np.unique(f[np.any(np.isin(f, list(ring)), axis=1)]))
print("肩點", c, seg[c])
rows = []
for a, b in E:
    if a in ring2 and b in ring2 and (a in ring or b in ring):
        r_ol = ol.get((a, b), ol.get((b, a)))
        lb = np.linalg.norm(v[a] - v[b]); l3 = np.linalg.norm(x[a] - x[b])
        rest = r_ol if r_ol is not None else lb
        rows.append((l3 / rest, a, b, seg[a], seg[b], r_ol, lb, l3))
for r in sorted(rows)[:25]:
    print(f"3D/原長 {r[0]:.2f}  {r[1]}({r[3]}) – {r[2]}({r[4]})  orig_lens {'-' if r[5] is None else round(r[5], 2)}  box {r[6]:.2f}  3D {r[7]:.2f}")

# 原長算出來的、肩點周圍每個三角形在肩點的內角
def rl(a, b):
    r = ol.get((a, b), ol.get((b, a)))
    return r if r is not None else float(np.linalg.norm(v[a] - v[b]))
tot = 0
for tri in f[np.any(f == c, axis=1)]:
    o = [t for t in tri if t != c]
    a, b = rl(c, o[0]), rl(c, o[1]); cc = rl(o[0], o[1])
    cosv = (a * a + b * b - cc * cc) / (2 * a * b)
    ang = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    tot += ang
    print(f"  三角 {c}-{o[0]}({seg[o[0]]})-{o[1]}({seg[o[1]]})  邊 {a:.2f} {b:.2f} {cc:.2f}  角 {ang:.0f}°" + ("  ← 三角不等式不成立" if cosv < -1 or cosv > 1 else ""))
print("肩點內角合計", round(tot), "°（平的布＝360）")

# 原長度量下每個頂點的內角合計（只算周圍一圈都是三角形的內部點；邊界點略過）
bnd = set(np.concatenate(igl.all_boundary_loop(f))) if len(igl.all_boundary_loop(f)) else set()
def angsum(p):
    t = 0
    for tri in f[np.any(f == p, axis=1)]:
        o = [q for q in tri if q != p]
        a, b, cc = rl(p, o[0]), rl(p, o[1]), rl(o[0], o[1])
        t += np.degrees(np.arccos(np.clip((a * a + b * b - cc * cc) / (2 * a * b), -1, 1)))
    return t
dist3 = np.linalg.norm(x - x[c], axis=1)
near = [p for p in np.flatnonzero(dist3 < 6) if p not in bnd]
print("肩點 6 cm 內的點（照 3D 距離排）：點 標籤 內角合計 離肩點")
for p in sorted(near, key=lambda q: dist3[q]):
    s_ = angsum(p)
    print(f"  {p:5d} {seg[p]:28s} {s_:6.0f}°  {dist3[p]:.1f} cm" + ("   <==" if abs(s_ - 360) > 15 else ""))
# 全件：偏離 360 超過 15° 的點有多少、在哪
allp = [p for p in range(len(v)) if p not in bnd]
dev = np.array([angsum(p) - 360 for p in allp])
big = np.array(allp)[np.abs(dev) > 15]
from collections import Counter
print("全件偏離 360° 超過 15° 的點：", len(big), Counter(seg[p].split(",")[0] if not seg[p].startswith("stitch") else "縫線點" for p in big))
