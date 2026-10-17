# 通用領深校正：5 種身體 × 14 件 AI 款式
import copy, sys, time
from pathlib import Path
import yaml
REPO = Path(r"D:\newfasion_VT_work\repo_ascii")
sys.path.insert(0, str(REPO / "tools" / "chatgarment"))
import design_to_mesh as dm   # 切到 GarmentCodeRC 根目錄
from assets.bodies.body_params import BodyParameters
from assets.garment_programs.meta_garment import MetaGarment
from neckline import fix_neck_depth
R = Path(r"C:\Users\Yuquan0230\OneDrive\桌面\newfasion_VT\results\chatgarment_ai")
bodies = ["mean_male", "mean_female", "mean_all", "m_smpl_average_A40", "f_smpl_average_A40"]
items = [(m, p.parent.name, p) for m in ["4bit_v2", "bf16_3090"] for p in sorted((R / m).glob("*/design_upper.yaml"))]
for bn in bodies:
    body = BodyParameters(str(dm.GC_ROOT / "assets" / "bodies" / f"{bn}.yaml"))
    print(f"\n== {bn}：neck_w {body['neck_w']:.1f} cm → 最淺 前 {0.39*body['neck_w']:.1f}、後 {0.10*body['neck_w']:.1f}", flush=True)
    for mode, g, p in items:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))["design"]
        d["meta"]["bottom"]["v"] = None; d["meta"]["wb"]["v"] = None
        t0 = time.time()
        try:
            r = fix_neck_depth(d, body, MetaGarment)
        except Exception as e:
            print(f"  {mode:9s} {g:26s} 失敗：{e}"); continue
        if r is None:
            print(f"  {mode:9s} {g:26s} 不適用"); continue
        tf, tb = r["目標深度 cm"]; gf, gb = r["實際深度 cm"]
        ok = "✔" if abs(tf - gf) < 0.2 and abs(tb - gb) < 0.2 else "✘"
        print(f"  {ok} {mode:9s} {g:26s} {d['collar']['f_collar']['v']:16s} 相對深度 {r['AI 相對深度（前, 後）']} → 目標 前{tf:5.1f} 後{tb:4.1f}，實際 前{gf:5.1f} 後{gb:4.1f}（{time.time()-t0:.1f} 秒）", flush=True)
