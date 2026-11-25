# -*- coding: utf-8 -*-
r"""把 spec_batch.py 照尺寸表縫好的各尺碼匯出給 engine（格式同 body_fit/export_neutral.py），並列進試穿頁清單。

每個尺碼一件：代號＝<衣服代號><尺碼>（z01S、z06XXS…），資料夾 <代號>_<英文名>。
  - OBJ：公分 → 公尺、往下移回 SMPL-X 原始座標（和 engine 的身體同座標）；三角形方向統一（sew.orient_faces）
  - 固定點：neck_shoulder_band（同 export_neutral 規則）；有帽子的另給 hood
    （spec_batch 沒留版片分區，帽子用「比頭頂低不到 5 cm、離頭 3 cm 內」挑，那一帶只有帽子）
  - 量測.json：照抄 spec_batch 的尺寸表與縫好量到的數字

跑法（在 newfashion_VT 根目錄，先跑過 spec_batch.py）：
    .venv\Scripts\python flows\size_chart\spec_batch_export.py [VTO_PROJ 根目錄] [--only z06M] [--suffix v2] [--textures]
      --textures：另外輸出縫線／細節貼圖（tools/texture/seam_trim.py、detail_trim.py）和 UV，OBJ 不動：
        <資料夾>_uv.json（每個面三個角的 UV，面的順序、角的順序和 OBJ 一樣）、_color.png、_normal.png；
        有立體零件的（扣子、帽繩）再加 parts.obj、parts_attach.json。格式見 docs/需求_試穿頁貼圖.md
      --only：只匯出這幾件（逗號分隔）；--suffix：代號後面加字，不能有底線（engine 用資料夾名第一個底線前的字當代號；和試穿頁上舊的同一件並排比較，不蓋掉舊的）
輸出：output/physics_export/size_chart/<代號>_<名稱>/；給了 VTO_PROJ 就複製到 garment/meshes/neutral/，
      並把這些衣服加進 Complete_System/garments.json（同代號的先拿掉再加，可重跑）
"""
import json
import shutil
import sys
from pathlib import Path

import igl
import numpy as np

HERE = Path(__file__).resolve().parent
BODY_FIT = HERE.parent / "body_fit"
sys.path.insert(0, str(BODY_FIT))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
import smplx_body  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import CONTACT_M, HOOD_CONTACT_M, HOOD_CROWN_M, RESOLUTION, band_pins  # noqa: E402
from sew import orient_faces  # noqa: E402

ROOT = sp.ROOT
BATCH = ROOT / "output" / "spec_to_pattern" / "_batch"
SOURCES = ROOT / "results" / "chatgarment_ai" / "bf16_z"
OUT = ROOT / "output" / "physics_export" / "size_chart"
ORDER = ["XXS", "XS", "S", "M", "L", "XL", "2XL", "XXL"]
NAMES = [("chest", "胸寬"), ("shoulder", "肩寬"), ("length", "衣長"), ("sleeve", "袖長"), ("hem", "下襬圍")]


def hood_pins(x, bv, bf):
    d = np.sqrt(igl.point_mesh_squared_distance(x, bv, bf)[0])
    top = float(bv[:, 1].max())
    idx = np.flatnonzero((x[:, 1] > top - HOOD_CROWN_M) & (d <= HOOD_CONTACT_M)).tolist()
    return idx or None


def write_obj(path, v, f):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines("v %.6f %.6f %.6f\n" % tuple(p) for p in v)
        fh.writelines("f %d %d %d\n" % tuple(t + 1) for t in f)


def summary(rec):
    t, g = rec["尺寸表"], rec["縫好量到"]
    return "、".join(f"{zh} {g[k]:.1f}／{t[k]:g}" for k, zh in NAMES if t.get(k) is not None and g.get(k) is not None)


def export_textures(g, sdir, dst, folder, flip, lift):
    """貼圖、UV、立體零件（--textures）。回傳說明。"""
    sys.path.insert(0, str(ROOT / "tools" / "texture"))
    import neck_trim
    import seam_trim
    import detail_trim
    s = sdir.name
    chat = sp.dm.OUT_ROOT / f"specb_{g}_{s}"
    mesh = sdir / f"{g}_{s}.obj"
    _, vt, fv, ft = neck_trim.read_obj(chat / f"specb_{g}_{s}_boxmesh.obj")
    x, f0 = igl.read_triangle_mesh(str(mesh))
    if len(fv) != len(f0) or not np.array_equal(np.sort(fv, 1), np.sort(f0, 1)):
        return "縫合資料夾和網格對不上，沒做貼圖（要重縫）"
    tmp = dst / "_tex"
    design = sdir / "design.yaml"
    if g in detail_trim.DETAILS:
        detail_trim.main(g, str(mesh), str(chat), str(tmp), str(design))
        maps = ("detail_color.png", "detail_normal.png")
    else:
        seam_trim.main(str(mesh), str(chat), str(tmp), 2048, str(design))
        maps = ("seam_color.png", "seam_normal.png")
    shutil.copy(tmp / maps[0], dst / f"{folder}_color.png")
    shutil.copy(tmp / maps[1], dst / f"{folder}_normal.png")
    ft = np.asarray(ft).copy()
    ft[flip] = ft[flip][:, ::-1]  # orient_faces 把這些面的頂點順序反過來了，UV 的角也要反
    (dst / f"{folder}_uv.json").write_text(json.dumps({
        "vt": np.round(vt, 6).tolist(), "ft": ft.tolist(),
        "note": "ft[k] 是 OBJ 第 k 個面三個角的 UV 編號（順序同 OBJ 的 f 行）；UV 原點左下、v 往上（同 OBJ 慣例）",
        "color": f"{folder}_color.png", "normal": f"{folder}_normal.png", "normal_space": "切線空間（OpenGL，+y 往上）",
    }), encoding="utf-8")
    note = "縫線" + ("＋細節" if g in detail_trim.DETAILS else "")
    if (tmp / "parts.obj").exists():
        lines = []
        for ln in (tmp / "parts.obj").read_text().splitlines():
            if ln.startswith("v "):
                v = [float(t) for t in ln.split()[1:4]]
                ln = "v %.6f %.6f %.6f" % (v[0], v[1] - lift, v[2])
            lines.append(ln)
        (dst / "parts.obj").write_text("\n".join(lines) + "\n")
        att = json.loads((tmp / "parts_attach.json").read_text(encoding="utf-8"))
        fl = set(int(k) for k in flip)
        for pt in att["parts"]:
            if "face" in pt and pt["face"] in fl:
                pt["bary"] = pt["bary"][::-1]
        att["garment"] = f"{folder}.obj"
        (dst / "parts_attach.json").write_text(json.dumps(att, ensure_ascii=False, indent=1), encoding="utf-8")
        note += f"、立體零件 {len(att['parts'])} 個"
    shutil.rmtree(tmp)
    return note


def main(vto=None, only=None, suffix="", textures=False):
    _, _, lift = prepare_smplx_body("neutral")
    bv, bf, weights = smplx_body.make_body("neutral")
    english = {d.name.split("_", 1)[0]: d.name.split("_", 1)[1] for d in SOURCES.iterdir() if d.is_dir()}
    entries = []
    for gdir in sorted(d for d in BATCH.iterdir() if d.is_dir() and d.name.startswith("z")):
        sizes = sorted((d for d in gdir.iterdir() if (d / "量測.json").exists()),
                       key=lambda d: ORDER.index(d.name) if d.name in ORDER else 99)
        for sdir in sizes:
            rec = json.loads((sdir / "量測.json").read_text(encoding="utf-8"))
            if only and f"{gdir.name}{sdir.name}" not in only:
                continue
            code = f"{gdir.name}{sdir.name}{suffix}"
            folder = f"{code}_{english[gdir.name]}"
            x, f = igl.read_triangle_mesh(str(sdir / f"{gdir.name}_{sdir.name}.obj"))
            f, flip = orient_faces(f)
            xm = x / 100.0 - [0.0, lift, 0.0]
            dst = OUT / folder
            dst.mkdir(parents=True, exist_ok=True)
            write_obj(dst / f"{folder}.obj", xm, f)
            tex_note = export_textures(gdir.name, sdir, dst, folder, flip, lift) if textures else None
            band = band_pins(xm, bv, bf, weights)
            hood = hood_pins(xm, bv, bf)
            inside = igl.fast_winding_number_for_meshes(bv, bf, xm) > 0.5
            (dst / f"{folder}_pins.json").write_text(json.dumps({
                "units": "m", "up_axis": "+y", "front_axis": "+z",
                "coordinates": "SMPL-X 原始座標（neutral、θ=0、β 全 0，和 engine 的身體相同，不用再平移對齊）",
                "pose": "T-pose（SMPL-X θ=0）",
                "n_vertices": int(len(xm)), "n_faces": int(len(f)), "has_uv": bool(tex_note and tex_note.startswith("縫線")), "resolution_cm": rec.get("頂點間距cm", RESOLUTION),
                "pins": {"neck_shoulder_band": band, **({"hood": hood} if hood else {})},
                "check": {"穿進身體的頂點%": round(100 * float(inside.mean()), 2), "固定點距離上限 m": CONTACT_M},
                "note": "照廠商尺寸表打版（flows/size_chart/spec_batch.py）；頂點編號 0 起算，順序和 OBJ 的 v 行相同",
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            shutil.copy(sdir / "量測.json", dst / "量測.json")
            name = f"{rec['衣服'].split('_', 1)[1]}・{sdir.name} 號" + (f"（{suffix.strip('_')}）" if suffix else "")
            entries.append({"code": code, "name": name, "mesh": f"garment/meshes/neutral/{folder}",
                            "fit_type": None, "has_size_chart": False,
                            "note": f"照廠商尺寸表 {sdir.name} 號打版；縫好量到／尺寸表（cm）：{summary(rec)}"})
            print(f"[{code}] {len(xm)} 頂點、固定點 {len(band)}" + (f"、帽子 {len(hood)}" if hood else "")
                  + f"、穿進身體 {100 * inside.mean():.2f}%、翻面 {len(flip)}" + (f"、貼圖：{tex_note}" if tex_note else ""), flush=True)

    if vto:
        vto = Path(vto)
        neutral = vto / "garment" / "meshes" / "neutral"
        for e in entries:
            src = OUT / Path(e["mesh"]).name
            dst = neutral / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        gpath = vto / "Complete_System" / "garments.json"
        garments = json.loads(gpath.read_text(encoding="utf-8"))
        codes = {e["code"] for e in entries}
        garments = [g for g in garments if g["code"] not in codes] + entries
        gpath.write_text(json.dumps(garments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"複製 {len(entries)} 件到 {neutral}，試穿清單共 {len(garments)} 件", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = {}
    for k in ("--only", "--suffix"):
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            del args[i:i + 2]
    tex = "--textures" in args
    if tex:
        args.remove("--textures")
    main(*args, only=set(opt["--only"].split(",")) if "--only" in opt else None, suffix=opt.get("--suffix", ""), textures=tex)
