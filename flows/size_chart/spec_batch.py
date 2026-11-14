# -*- coding: utf-8 -*-
r"""照尺寸表打版的整批版：網路衣服的每一件、每一個尺碼都打版、縫到 SMPL-X 中性身體上，
再量縫好的 3D 衣服，和尺寸表（廠商平放量的 cm）比差多少。

款式：每件借一件款式相近的 AI 結果（TEMPLATES），套上領深校正、合身款改一般款；
     胸寬、衣長、袖長照尺寸表（spec_to_pattern.fit，2D 版片上量到差 0.2 cm 內），肩寬不照（往外推會縫壞）。
3D 量法（縫好、T-pose、沒有重力，單位 cm）。尺寸表是平放「沿著布」量的，所以 3D 也沿著布面量：
  胸寬  腋下往下 2.5 cm 那一圈衣身的周長 ÷ 2（等於平放時的寬度）
  衣長  後片最高點（肩頸點，尺寸表也從這裡量）到後中心下襬，在布面上的最短路徑（測地線）
  袖長  肩點（衣身在袖籠上半最外側的點）到袖口最遠點，在布面上的最短路徑，左右平均
  肩寬  左右肩點的距離
  下襬圍 下襬那一圈的長度（尺寸表有才比）
  另外記「袖口伸到多遠」＝肩寬÷2＋袖長（日本尺寸表的「裄丈」），看整隻袖子跟真的衣服差多少

跑法（在 repo 根目錄；這台筆電要從英文路徑副本跑，libigl 開不了中文路徑）：
    .venv\Scripts\python flows\size_chart\spec_batch.py [衣服代號…，預設全部]     一件一件做（可以開好幾個程序分開跑）
    .venv\Scripts\python flows\size_chart\spec_batch.py --remeasure                  量法改了：用縫好的網格重量，不重縫
    .venv\Scripts\python flows\size_chart\spec_batch.py --summary                    把做好的合成 總表.json（網頁用）
輸出：output/spec_to_pattern/_batch/<代號>/<尺碼>/（縫好的 .obj、量測.json）、_batch/總表.json
"""
import base64
import copy
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import igl
import numpy as np
import yaml
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
BODY_FIT = HERE.parent / "body_fit"
sys.path.insert(0, str(BODY_FIT))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import build, load_design  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from sew import unfold  # noqa: E402

DATA = HERE / "網路衣服_尺寸資料"
AI = sp.ROOT / "results" / "chatgarment_ai"
OUT = sp.ROOT / "output" / "spec_to_pattern" / "_batch"
# 每件網路衣服借哪一件 AI 結果的款式（領型、袖型、帽子、前襟…）
TEMPLATES = {
    "z01_帽T": "bf16_i/i46_blue_hoodie",
    "z02_襯衫": "bf16_i/i44_green_button_down_shirt",
    "z03_針織衫": "bf16_i/i02_brown_crew_sweatshirt",
    "z04_POLO": "bf16_i/i40_charcoal_polo",
    "z05_開襟衫": "bf16_h/h08_frenchblue_cardigan",
    "z06_長袖T": "bf16_i/i10_red_crew_long_sleeve_tee",
    "z07_U領長袖T": "bf16_i/i24_black_square_neck_long_sleeve_top",
}
NAMES = {"chest": "胸寬", "shoulder": "肩寬", "length": "衣長", "sleeve": "袖長", "hem": "下襬圍"}
CHEST_BELOW_ARMPIT = 2.5


def girth(pts):
    """一圈點（xz 平面）的周長，用凸包（平放量的是外圍，凹進去的皺褶不算）。"""
    return float(ConvexHull(pts).area) if len(pts) >= 3 else float("nan")


def geodesic(x, f, src, dst):
    """網格表面上 src 到 dst（頂點編號）的最短路徑長度（cm），等於沿著布量。
    用熱擴散法（heat geodesic，時間參數取平均邊長平方）；igl 的 exact_geodesic 在這版會回傳空陣列。"""
    x = np.asarray(x, float)
    f = np.asarray(f, np.int64)
    e = igl.edges(f)
    t = float(np.mean(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1))) ** 2
    d = igl.heat_geodesic(x, f, t, np.array([src], np.int64))
    return float(d[dst])


def measure_3d(x, f, seg):
    """縫好的衣服（cm、腳底在 0、T-pose）量尺寸表的幾個欄位。seg：每個頂點的版片名稱。"""
    seg = np.asarray(seg)
    torso = np.array(["torso" in s for s in seg])
    back = np.array(["btorso" in s for s in seg])
    sleeve = np.array(["sleeve" in s or "cuff" in s for s in seg])
    out = {}
    # 腋下：袖子靠身體那一端最低的點
    near = sleeve & (np.abs(x[:, 0]) < np.abs(x[torso, 0]).max() + 4)
    armpit = x[near, 1].min() if near.any() else x[torso, 1].max() - 20
    y0 = armpit - CHEST_BELOW_ARMPIT
    band = torso & (np.abs(x[:, 1] - y0) < 1.0)
    out["chest"] = girth(x[band][:, [0, 2]]) / 2
    # 衣長：後片最高點（肩頸點）→ 後中心（|x| < 2 cm）最低點
    b = np.flatnonzero(back)
    center = np.flatnonzero(back & (np.abs(x[:, 0]) < 2.0))
    out["length"] = geodesic(x, f, b[x[b, 1].argmax()], center[x[center, 1].argmin()]) if len(center) else float("nan")
    upper = torso & (x[:, 1] > armpit + 3)
    sides = []
    for sign in (1, -1):
        t = np.flatnonzero(upper & (sign * x[:, 0] > 0))
        s = np.flatnonzero(sleeve & (sign * x[:, 0] > 0))
        if len(t) and len(s):
            shoulder_pt = t[np.abs(x[t, 0]).argmax()]
            cuff_pt = s[np.abs(x[s, 0]).argmax()]
            sides.append((abs(x[shoulder_pt, 0]), geodesic(x, f, shoulder_pt, cuff_pt)))
    out["shoulder"] = float(sum(s[0] for s in sides)) if len(sides) == 2 else float("nan")
    out["sleeve"] = float(np.mean([s[1] for s in sides])) if sides else float("nan")
    loops = igl.all_boundary_loop(f)
    hem = min(loops, key=lambda L: x[L, 1].mean())
    p = x[hem]
    out["hem"] = float(np.linalg.norm(p - np.roll(p, 1, 0), axis=1).sum())
    return out


def code(gid):
    """z06_長袖T → z06（輸出路徑只用英文代號）"""
    return gid.split("_")[0]


def template_for(gid, body):
    design = load_design(AI / TEMPLATES[gid])
    if design["meta"]["upper"]["v"] == "FittedShirt":  # 合身款不看衣長參數
        design["meta"]["upper"]["v"] = "Shirt"
    with contextlib.redirect_stdout(io.StringIO()):
        fix_neck_depth(design, body, MetaGarment)
    return design


def pack_mesh(x, f):
    """網頁用：頂點量化到 mm（int16）、面（uint16），都轉成 base64。"""
    q = np.round(x * 10).astype(np.int16)
    return {"v": base64.b64encode(q.tobytes()).decode(), "f": base64.b64encode(f.astype(np.uint16).tobytes()).decode(),
            "nv": int(len(x)), "nf": int(len(f))}


def run(gid, body_obj, body_yaml):
    info = json.loads((DATA / gid / "尺寸.json").read_text(encoding="utf-8"))
    sp.BASE_BODY = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body_mesh = (bv * 100, bf)
    template = template_for(gid, sp.BASE_BODY)
    keys = dict(sp.KEYS, **{"すそ周り": "hem"})
    for size, vals in info["sizes"].items():
        dst = OUT / code(gid) / size
        if (dst / "量測.json").exists():
            print(f"[{gid} {size}] 已經做過", flush=True)
            continue
        t = time.time()
        table = {keys[c]: float(v) for c, v in zip(info["cols"], vals) if c in keys and not isinstance(v, str)}
        target = {k: v for k, v in table.items() if k in ("chest", "length", "sleeve")}
        with contextlib.redirect_stdout(io.StringIO()):
            design, _, m2d, hist, params = sp.fit(target, template)
        name = f"specb_{gid.split('_')[0]}_{size}"
        folder, x, f, _, _ = build(design, name, body_yaml, body_mesh)
        x, unf = unfold(x, f, body=body_mesh, steps=3, max_ring=0)
        seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
        m3d = measure_3d(x, f, seg)
        dst.mkdir(parents=True, exist_ok=True)
        igl.write_triangle_mesh(str(dst / f"{code(gid)}_{size}.obj"), x, f)
        (dst / "design.yaml").write_text(yaml.safe_dump(json.loads(json.dumps({"design": design}, default=float)),
                                                        allow_unicode=True), encoding="utf-8")
        rec = {"衣服": gid, "尺碼": size, "尺寸表": table, "版片量到": {k: round(v, 1) for k, v in m2d.items()},
               "縫好量到": {k: round(v, 1) for k, v in m3d.items()}, "參數": {k: round(v, 3) for k, v in params.items()},
               "照尺寸表的欄位": sorted(target), "頂點數": int(len(x)), "尖角攤平": unf, "秒": round(time.time() - t)}
        (dst / "量測.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        m3d["reach"] = m3d["shoulder"] / 2 + m3d["sleeve"]
        rec["縫好量到"]["reach"] = round(m3d["reach"], 1)
        (dst / "量測.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{gid} {size}] {rec['秒']} 秒｜" + "  ".join(
            f"{NAMES[k]} {m3d[k]:.1f}/{table[k]}" for k in table), flush=True)


def remeasure():
    """量法改了時用：讀縫好的網格和縫合資料夾的分區檔，重算「縫好量到」，不用重縫。"""
    for d in sorted(OUT.glob("*/*/量測.json")):
        rec = json.loads(d.read_text(encoding="utf-8"))
        name = f"specb_{code(rec['衣服'])}_{rec['尺碼']}"
        seg = (sp.dm.OUT_ROOT / name / f"{name}_sim_segmentation.txt").read_text().splitlines()
        x, f = igl.read_triangle_mesh(str(d.parent / f"{code(rec['衣服'])}_{rec['尺碼']}.obj"))
        m3d = measure_3d(x, f, seg)
        m3d["reach"] = m3d["shoulder"] / 2 + m3d["sleeve"]
        rec["縫好量到"] = {k: round(v, 1) for k, v in m3d.items()}
        d.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{rec['衣服']} {rec['尺碼']}] " + "  ".join(f"{NAMES.get(k, k)} {m3d[k]:.1f}/{rec['尺寸表'].get(k, '-')}"
                                                         for k in ("chest", "shoulder", "length", "sleeve", "hem")), flush=True)


def summary():
    """把每件、每個尺碼的量測合成一份（網頁用），附網格。"""
    items = []
    for gid in TEMPLATES:
        info = json.loads((DATA / gid / "尺寸.json").read_text(encoding="utf-8"))
        for size in info["sizes"]:
            d = OUT / code(gid) / size
            if not (d / "量測.json").exists():
                continue
            rec = json.loads((d / "量測.json").read_text(encoding="utf-8"))
            x, f = igl.read_triangle_mesh(str(d / f"{code(gid)}_{size}.obj"))
            rec["mesh"] = pack_mesh(x, f)
            items.append(rec)
        meta = {k: info.get(k) for k in ("name", "brand", "type", "features", "material", "url")}
        for r in items:
            if r["衣服"] == gid:
                r["商品"] = meta
                r["款式來源"] = TEMPLATES[gid]
    (OUT / "總表.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} 件 → {OUT / '總表.json'}")


def main(*args):
    if "--summary" in args:
        return summary()
    if "--remeasure" in args:
        return remeasure()
    sp.dm.ensure_system_json()
    body_obj, body_yaml, _ = prepare_smplx_body("neutral")
    for gid in (args or TEMPLATES):
        run(gid, body_obj, body_yaml)


if __name__ == "__main__":
    main(*sys.argv[1:])
