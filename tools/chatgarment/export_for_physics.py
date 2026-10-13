# -*- coding: utf-8 -*-
"""
產生交給紀泓宇物理模擬用的衣服網格（對應《組員確認清單》B-1、B-2）：

  - 姿勢：T-pose（GarmentCode 的 arm_pose_angle = 0，和 SMPL-X θ=0 一致）
  - 單位：公尺；三角形面；保留 UV
  - 頂點數：用 --resolution（頂點間距，cm）控制，2.0 約 2,000 頂點
  - 固定點（JSON）：neck_shoulder_band＝和紀泓宇相同規則的「領口到肩線一圈」（建議用這組），
    另附領口一圈、肩線縫線兩組；只重算固定點：--repin [名稱…]

紀泓宇的 SMPL-X 人體還沒交過來前，先用 GarmentCode 附的 SMPL 女性模板（T-pose）測試；
之後用 --body-obj／--body-yaml 換成他的人體和量測值。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\export_for_physics.py --resolution 2.0
    .venv\\Scripts\\python tools\\chatgarment\\export_for_physics.py --design 某件的design.yaml --name 名稱

輸出：output/physics_export/<名稱>/<名稱>.obj、<名稱>_pins.json、預覽用的 box mesh 與縫合結果
"""
import argparse
import json
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

CWD = Path.cwd()  # design_to_mesh 匯入時會切換目錄，先記下來，命令列給的相對路徑以這裡為準
sys.path.insert(0, str(Path(__file__).parent))
import design_to_mesh as dm  # noqa: E402  （匯入時會切到 GarmentCodeRC 根目錄）
import sew  # noqa: E402

from pygarment.meshgen import boxmeshgen  # noqa: E402

# GarmentCode 的 bug：邊太短時要印警告，但警告訊息用到 Edge 沒有的 name 屬性而當掉（翻領在低解析度會遇到）。
# 不改第三方程式碼，在這裡補一個預設值讓警告能正常印出。
boxmeshgen.Edge.name = "edge"

REPO = dm.REPO
OUT = REPO / "output" / "physics_export"
TEMPLATE = dm.GC_ROOT / "assets" / "bodies" / "f_smpl_template"


def prepare_tpose_body():
    """GarmentCode 附的 SMPL 女性模板：網格本身是 T-pose，單位公尺、原點在骨盆。
    轉成本專案慣例（腳底在 y=0、單位公尺），尺寸檔的手臂角度改成 0（T-pose）。"""
    folder = OUT / "_bodies"
    folder.mkdir(parents=True, exist_ok=True)
    obj, yml = folder / "tpose_smpl_female.obj", folder / "tpose_smpl_female.yaml"
    v, f = igl.read_triangle_mesh(str(TEMPLATE.with_suffix(".obj")))
    v = v.copy()
    v[:, 1] -= v[:, 1].min()
    igl.write_triangle_mesh(str(obj), v, f)
    body = yaml.safe_load(TEMPLATE.with_suffix(".yaml").read_text())
    body["body"]["arm_pose_angle"] = 0
    yml.write_text(yaml.safe_dump(body), encoding="utf-8")
    return obj, yml


def pin_vertices(folder, name):
    """領口一圈（GarmentCode 標的 *_collar）＋肩線（前後片之間、位置最高的那條縫線）。"""
    labels = yaml.safe_load((folder / f"{name}_vertex_labels.yaml").read_text())
    neckline = sorted({i for key, ids in labels.items() if "collar" in key for i in ids})

    seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
    spec = json.loads((folder / f"{name}_specification.json").read_text())
    v, _ = igl.read_triangle_mesh(str(folder / f"{name}_boxmesh.obj"))
    shoulder = set()
    for side in ("left", "right"):
        seams = []
        for k, stitch in enumerate(spec["pattern"]["stitches"]):
            panels = {e["panel"] for e in stitch if isinstance(e, dict)}  # 有些縫線多一個方向標記字串
            if panels == {f"{side}_ftorso", f"{side}_btorso"}:
                ids = [i for i, s in enumerate(seg) if f"stitch_{k}" in s.split(",")]
                if ids:
                    seams.append((v[ids, 1].mean(), ids))
        if seams:
            shoulder.update(max(seams)[1])  # 前後片之間有肩線和側縫，肩線比較高
    return neckline, sorted(shoulder)


BAND_PARTS = ("neck", "leftShoulder", "rightShoulder")  # SMPL 分區名稱：頸、左右鎖骨（關節 12、13、14）


def band_pins(x, body_obj, contact_cm=1.0):
    """照紀泓宇的做法選「領口到肩線一圈」：衣服頂點最近的身體面由頸與左右鎖骨主導。
    主導＝那個面的三個角在 SMPL 蒙皮分區（smpl_vert_segmentation.json）都屬於頸或鎖骨；
    再要求離身體 contact_cm 內（貼在身上的才綁，翻領、帽子這種浮起來的不綁）。
    x：衣服頂點（公尺）。身體要是 SMPL 拓撲（6,890 頂點）才能用這份分區表。"""
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    seg = json.loads((dm.GC_ROOT / "assets" / "bodies" / "smpl_vert_segmentation.json").read_text())
    if len(bv) != 6890:
        raise ValueError(f"身體有 {len(bv)} 個頂點，不是 SMPL 拓撲，不能用 SMPL 分區表")
    in_band = np.zeros(len(bv), bool)
    for part in BAND_PARTS:
        in_band[seg[part]] = True
    d2, face, _ = igl.point_mesh_squared_distance(x, bv, bf)
    keep = in_band[bf[face]].all(axis=1) & (np.sqrt(d2) * 100 < contact_cm)
    return np.flatnonzero(keep).tolist()


def write_pins(dst, name, x, f, resolution, body_obj, neckline, shoulder):
    band = band_pins(x, body_obj)
    info = {
        "units": "m",
        "pose": "T-pose（GarmentCode arm_pose_angle=0，對應 SMPL-X θ=0）",
        "up_axis": "+y", "front_axis": "+z", "feet_at_y": 0.0,
        "n_vertices": int(len(x)), "n_faces": int(len(f)),
        "has_uv": True, "skinning_weights": None,
        "resolution_cm": resolution,
        "body_obj": str(Path(body_obj).relative_to(REPO)),
        "pins": {"neck_shoulder_band": band,  # 建議用這組：和紀泓宇自動推的規則相同
                 "neckline": neckline, "shoulder": shoulder,
                 "neckline_and_shoulder": sorted(set(neckline) | set(shoulder))},
        "note": "頂點編號是 0 起算，順序和 OBJ 的 v 行相同；neck_shoulder_band＝最近身體面由頸與左右鎖骨主導、且離身體 1 cm 內",
    }
    (dst / f"{name}_pins.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {dst / (name + '.obj')}：{len(x)} 頂點、{len(f)} 面；"
          f"固定點 頸肩帶 {len(band)}（領口 {len(neckline)}＋肩線 {len(shoulder)}）")


def repin(name):
    """不重新打版縫合，只用現有的輸出重算固定點。"""
    dst = OUT / name
    old = json.loads((dst / f"{name}_pins.json").read_text(encoding="utf-8"))
    x, f = igl.read_triangle_mesh(str(dst / f"{name}.obj"))
    write_pins(dst, name, x, f, old["resolution_cm"], REPO / old["body_obj"],
               old["pins"]["neckline"], old["pins"]["shoulder"])


def export(design, name, body_obj, body_yaml, resolution):
    _, box, paths = dm.generate(design, body_yaml, name, resolution=resolution)
    folder = paths.g_box_mesh.parent
    sew.main(folder, body_obj)

    sewn = folder / f"{name}_sewn.obj"
    x, f = igl.read_triangle_mesh(str(sewn))
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    sew.write_obj_with_new_positions(sewn, dst / f"{name}.obj", x / 100.0)  # 公分 → 公尺
    for extra in ("_material.mtl", "_texture_fabric.png"):
        src = folder / f"{name}{extra}"
        if src.exists():
            (dst / src.name).write_bytes(src.read_bytes())

    neckline, shoulder = pin_vertices(folder, name)
    write_pins(dst, name, x / 100.0, f, resolution, body_obj, neckline, shoulder)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=str(dm.DESIGN), help="design.yaml，預設是 ChatGarment 範例 T 恤")
    ap.add_argument("--name", default="tshirt")
    ap.add_argument("--resolution", type=float, default=2.0, help="頂點間距（cm）")
    ap.add_argument("--body-obj", help="T-pose 身體網格（公尺、腳底在 0）；不給就用 SMPL 女性模板")
    ap.add_argument("--body-yaml", help="對應的 GarmentCode 身體尺寸檔（arm_pose_angle 要是 0）")
    ap.add_argument("--repin", nargs="*", metavar="名稱",
                    help="只重算固定點（不重新縫合）；不給名稱就全部 output/physics_export 底下的")
    args = ap.parse_args()

    if args.repin is not None:
        for name in args.repin or sorted(p.name for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_")):
            repin(name)
        return

    dm.ensure_system_json()
    if args.body_obj:
        body_obj, body_yaml = CWD / args.body_obj, CWD / args.body_yaml
    else:
        body_obj, body_yaml = prepare_tpose_body()
    design = yaml.safe_load((CWD / args.design).read_text())["design"]
    design["meta"]["bottom"]["v"] = None  # 只做上衣
    design["meta"]["wb"]["v"] = None
    export(design, f"{args.name}_r{args.resolution:g}", body_obj, body_yaml, args.resolution)


if __name__ == "__main__":
    main()
