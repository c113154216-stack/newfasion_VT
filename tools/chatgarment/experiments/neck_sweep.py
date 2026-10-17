# 掃描領口參數：寬度、前領深 → 2D 版片的領口周長（不縫合，只生成 box mesh）
import copy, pickle, sys, yaml, io, contextlib
from pathlib import Path
import numpy as np, igl
REPO = Path(r"D:\newfasion_VT_work\repo_ascii")
sys.path.insert(0, str(REPO / "tools" / "chatgarment"))
import design_to_mesh as dm
from sew import mesh_edges, rest_lengths
from sew_experiment import openings
dm.ensure_system_json()
src = Path(r"C:\Users\Yuquan0230\OneDrive\桌面\newfasion_VT\results\chatgarment_ai\4bit_v2\g06_red_tee\design_upper.yaml")
base = yaml.safe_load(src.read_text(encoding="utf-8"))["design"]
base["meta"]["bottom"]["v"] = None; base["meta"]["wb"]["v"] = None
body = dm.GC_ROOT / "assets" / "bodies" / "mean_male.yaml"
for w, fd, bd in [(0.197, 0.15, 0.084), (0.197, 0.0, 0.084), (0.197, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (-0.2, 0.1, 0.0)]:
    d = copy.deepcopy(base); d["collar"]["width"]["v"] = w; d["collar"]["fc_depth"]["v"] = fd; d["collar"]["bc_depth"]["v"] = bd
    name = f"necksweep_w{w}_fd{fd}_bd{bd}"
    with contextlib.redirect_stdout(io.StringIO()):
        _, box, paths = dm.generate(d, body, name)
    v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
    e = mesh_edges(f); rest = rest_lengths(v, e, pickle.loads(paths.g_orig_edge_len.read_bytes()))
    neck = openings(v, f, e)[0][0]
    nb = v[e[neck]].reshape(-1, 3)
    import json
    pan = json.load(open(next(Path(paths.in_g_spec).parent.glob("*_specification.json"))))["pattern"]["panels"]
    fv = np.array(pan["left_ftorso"]["vertices"]); bv = np.array(pan["left_btorso"]["vertices"])
    neck_y = fv[:, 1].max(); half_w = fv[fv[:, 1].argmax(), 0]
    print(f"寬度 {w:+.2f}、前領深 {fd:.2f}、後領深 {bd:.3f} → 領口寬 {2*half_w:.1f} cm、前深 {neck_y - fv[-1,1]:.1f} cm、後深 {bv[:,1].max() - bv[1,1]:.1f} cm → 版片領口周長 {rest[neck].sum():.1f} cm", flush=True)
