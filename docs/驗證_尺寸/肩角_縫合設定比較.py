"""肩角被壓的原因：同一件 box mesh 用不同縫合設定各縫一次，量肩寬、袖長、肩點附近的邊長比。
用法：python sewtry.py z06 M 變體名...
"""
import sys, json, pickle, time
import numpy as np, igl
from pathlib import Path
sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\size_chart")
sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\body_fit")
import spec_batch as sb  # noqa  （切目錄、載入 body_fit）
import spec_measure as sm
from sew import mesh_edges, rest_lengths, sew, unfold
from export_neutral import SEW
from export_for_physics import prepare_smplx_body

R = Path(r"C:\Users\User\Desktop\newfashion_VT\output\chatgarment")
B = Path(r"C:\Users\User\Desktop\newfashion_VT\output\spec_to_pattern\_batch")
OUT = Path(__file__).resolve().parent / "sewtry_out"
OUT.mkdir(exist_ok=True)

VARIANTS = {
    "現在": dict(SEW),
    "沒有彎曲": {**SEW, "bend": 0.0},
    "彎曲0.2": {**SEW, "bend": 0.2},
    "彎曲3": {**SEW, "bend": 3.0},
    "壓短更硬": {**SEW, "stretch_weight": 1000.0},
    "1200輪": {**SEW, "iterations": 1200},
    "沒有身體": {**SEW, "_nobody": True},
}


def corner_ratio(x, f, seg, rest, E, c):
    d = np.linalg.norm(x - x[c], axis=1)
    mid = (d[E[:, 0]] + d[E[:, 1]]) / 2
    r = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1) / rest
    m = (mid < 2.5) & (rest > 1e-6)
    return float(np.median(r[m])), float(np.percentile(r[m], 10))


def main(g, s, *names):
    name = f"specb_{g}_{s}"
    d = R / name
    rec = json.loads((B / g / s / "量測.json").read_text(encoding="utf-8"))
    lm = rec["量測點"]
    v, f = igl.read_triangle_mesh(str(d / f"{name}_boxmesh.obj"))
    seg = (d / f"{name}_sim_segmentation.txt").read_text().splitlines()
    E = mesh_edges(f)
    rest = rest_lengths(v, E, pickle.loads((d / f"{name}_orig_lens.pickle").read_bytes()))
    body_obj, _, _ = prepare_smplx_body("neutral")
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body = (bv * 100, bf)
    pat = rec["版片量到"]
    for nm in names or VARIANTS:
        t = time.time()
        kw = dict(VARIANTS[nm]); nob = kw.pop("_nobody", False)
        x0 = sew(v, E, rest, body=None if nob else body, **kw)
        from sew import folded_faces
        fo = folded_faces(x0, f)
        ffo = f[fo]
        for lbl, xx in (("攤平前", x0),):
            cr0 = [corner_ratio(xx, f, seg, rest, E, lm["肩點"][k]) for k in ("left", "right")]
            near = [int(np.sum(np.linalg.norm(xx[ffo].mean(1) - xx[lm["肩點"][k]], axis=1) < 4)) for k in ("left", "right")]
            print(f"   {lbl}：肩寬 {sm.shoulder(xx, f, seg, lm):.1f} 袖長 {sm.sleeve(xx, f, seg, lm):.1f} 肩點邊長比 {np.mean([c[0] for c in cr0]):.2f}"
                  f"  翻折面 {len(fo)}，在肩點 4 cm 內 {near}", flush=True)
        x, _ = unfold(x0, f, body=None if nob else body, steps=3, max_ring=0)
        sh = sm.shoulder(x, f, seg, lm)
        sl = sm.sleeve(x, f, seg, lm)
        ln = sm.length(x, f, seg, lm)
        cr = [corner_ratio(x, f, seg, rest, E, lm["肩點"][k]) for k in ("left", "right")]
        np.save(OUT / f"{g}_{s}_{nm}.npy", x)
        print(f"{g} {s} {nm:6s} 肩寬 {sh:.1f}（版 {pat['shoulder']}，{sh - pat['shoulder']:+.1f}）"
              f" 袖長 {sl:.1f}（{sl - pat['sleeve']:+.1f}） 衣長 {ln:.1f}（{ln - pat['length']:+.1f}）"
              f" 肩點 2.5 cm 內邊長比 中位 {np.mean([c[0] for c in cr]):.2f} 10% {np.mean([c[1] for c in cr]):.2f}"
              f"  {time.time() - t:.0f} 秒", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
