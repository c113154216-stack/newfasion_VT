# 紅 T 試算：用公分定領口目標，反推 GarmentCode 參數 → 打版 → 縫合（新設定）→ 放進紅 T 的實驗資料夾一起比較
import copy, io, contextlib, json, pickle, sys, time
from pathlib import Path
import numpy as np, yaml, igl
REPO = Path(r"D:\newfasion_VT_work\repo_ascii")
sys.path.insert(0, str(REPO / "tools" / "chatgarment"))
import design_to_mesh as dm
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions
from sew_experiment import openings, report
dm.ensure_system_json()

src = Path(r"C:\Users\Yuquan0230\OneDrive\桌面\newfasion_VT\results\chatgarment_ai\4bit_v2\g06_red_tee\design_upper.yaml")
design = yaml.safe_load(src.read_text(encoding="utf-8"))["design"]
design["meta"]["bottom"]["v"] = None; design["meta"]["wb"]["v"] = None
body_file = dm.GC_ROOT / "assets" / "bodies" / "mean_male.yaml"
B = yaml.safe_load(body_file.read_text())["body"]
bust_line = (2 / 3) * B["vert_bust_line"] + (1 / 3) * B["bust_line"]
tg = np.tan(np.deg2rad(B["shoulder_incl"]))

MODE = sys.argv[1] if len(sys.argv) > 1 else "neckfix"
design["collar"]["width"]["v"] = 0.0           # 開口寬 ＝ 身體頸寬
if MODE == "neckfix":   # 用公分定目標，反推參數
    TARGET_FRONT, TARGET_BACK = 8.0, 2.0
    half_w = B["neck_w"] / 2
    # 肩斜補正 ＝ tan(肩斜) × (軀幹版片半寬 − 開口半寬)；版片半寬從現在的版片讀（前 27.5、後 26.8）
    f_half, b_half = 27.5, 26.8
    adj_f, adj_b = tg * (f_half - half_w), tg * (b_half - half_w)
    design["collar"]["fc_depth"]["v"] = (TARGET_FRONT - adj_f) / bust_line
    design["collar"]["bc_depth"]["v"] = (TARGET_BACK - adj_b) / bust_line
    label = "前 8 cm、後 2 cm（反推）"
else:                    # 簡單版：只把參數調到 0 附近，不用負值
    adj_f = adj_b = float("nan")
    design["collar"]["fc_depth"]["v"] = 0.05
    design["collar"]["bc_depth"]["v"] = 0.0
    label = "寬 0、前 0.05、後 0"
print(f"胸線高 {bust_line:.1f}、肩斜補正 前 {adj_f:.1f}／後 {adj_b:.1f} → fc_depth {design['collar']['fc_depth']['v']:.3f}、bc_depth {design['collar']['bc_depth']['v']:.3f}")

name = f"ai4bit_v2_g06_red_tee_{MODE}"
with contextlib.redirect_stdout(io.StringIO()):
    _, box, paths = dm.generate(design, body_file, name)
spec = json.load(open(paths.in_g_spec))["pattern"]["panels"]
fv = np.array(spec["left_ftorso"]["vertices"]); bv = np.array(spec["left_btorso"]["vertices"])
print(f"版片：前深 {fv[:,1].max() - fv[-1,1]:.1f} cm、後深 {bv[:,1].max() - bv[1,1]:.1f} cm")

v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
edges = mesh_edges(f); rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
neck = openings(v, f, edges)[0][0]
bvtx, bf = igl.read_triangle_mesh(str(dm.GC_ROOT / "assets" / "bodies" / "mean_male.obj")); body = (bvtx * 100, bf)
t0 = time.time()
params = {"stretch_limit": 0.02, "stretch_weight": 100.0, "bend": 1.0}
x = sew(v, edges, rest, body=body, **params)
r = {"參數": {**params, "領口": label}, "秒": round(time.time() - t0), **report(x, f, edges, rest, body, neck)}
sel = (np.abs(x[:, 0]) < 1.5) & (x[:, 2] > 0)
r["前中心最高 y cm"] = round(float(x[sel, 1].max()), 1); r["前中心最低 y cm"] = round(float(x[sel, 1].min()), 1)
print(json.dumps(r, ensure_ascii=False))

# 放進紅 T 的實驗資料夾，用 sew_variants.html 跟其他設定並排看
red = REPO / "output" / "chatgarment" / "ai4bit_v2_g06_red_tee"
write_obj_with_new_positions(paths.g_box_mesh, red / f"ai4bit_v2_g06_red_tee_sewn_{MODE}.obj", x)
j = red / "sew_experiment.json"; allr = json.loads(j.read_text(encoding="utf-8")); allr[MODE] = r
j.write_text(json.dumps(allr, ensure_ascii=False, indent=2), encoding="utf-8")
