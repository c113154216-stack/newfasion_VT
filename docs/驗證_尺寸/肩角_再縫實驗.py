"""從整批第一次縫好的位置重做：攤平（統一方向的面）→ [再縫 → 攤平] × N，看肩寬每輪的變化。"""
import sys, json, pickle, time
import numpy as np, igl
from pathlib import Path
sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\size_chart")
sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\body_fit")
import spec_batch as sb  # noqa
import spec_measure as sm
from sew import mesh_edges, rest_lengths, sew, unfold, folded_faces, orient_faces
from export_neutral import SEW
from export_for_physics import prepare_smplx_body
R = Path(r"C:\Users\User\Desktop\newfashion_VT\output\chatgarment"); B = Path(r"C:\Users\User\Desktop\newfashion_VT\output\spec_to_pattern\_batch")
body_obj, _, _ = prepare_smplx_body("neutral"); bv, bf = igl.read_triangle_mesh(str(body_obj)); body = (bv * 100, bf)
for item in sys.argv[1:]:
    g, s = item.split("/")
    name = f"specb_{g}_{s}"; d = R / name
    rec = json.loads((B / g / s / "量測.json").read_text(encoding="utf-8")); lm = rec["量測點"]; pat = rec["版片量到"]
    v, f = igl.read_triangle_mesh(str(d / f"{name}_boxmesh.obj"))
    seg = np.array((d / f"{name}_sim_segmentation.txt").read_text().splitlines())
    fo, _ = orient_faces(f)
    E = mesh_edges(f); rest = rest_lengths(v, E, pickle.loads((d / f"{name}_orig_lens.pickle").read_bytes()))
    t = time.time()
    x = sew(v, E, rest, body=body, **SEW)

    def rep(tag, x):
        m = sb.measure_3d(x, f, seg, lm)
        print(f"{g} {s} {tag:12s} 肩寬 {m['shoulder']:.1f}({m['shoulder'] - pat['shoulder']:+.1f}) 袖長 {m['sleeve']:.1f}({m['sleeve'] - pat['sleeve']:+.1f})"
              f" 衣長 {m['length']:.1f}({m['length'] - pat['length']:+.1f}) 胸寬 {m['chest']:.1f}({m['chest'] - pat['chest']:+.1f}) 翻折 {len(folded_faces(x, fo))}  {time.time() - t:.0f}s", flush=True)

    rep("縫好", x)
    x, _ = unfold(x, fo, body=body, steps=3, max_ring=0)
    rep("攤平", x)
    for k in range(3):
        x = sew(x, E, rest, body=body, **{**SEW, "bend": 3.0, "iterations": 200})
        x, _ = unfold(x, fo, body=body, steps=3, max_ring=0)
        rep(f"再縫{k + 1}", x)
