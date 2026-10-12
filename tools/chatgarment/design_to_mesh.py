# -*- coding: utf-8 -*-
"""
驗證「款式（ChatGarment 輸出的 design.yaml）＋ 身體尺寸 → 3D 衣服網格」這一段。

ChatGarment 的 AI 只輸出款式參數（design.yaml），衣服大小由另外給的身體尺寸決定。
這支程式不跑 AI，直接拿 ChatGarment 附的範例 design.yaml，
配上幾組不同的身體，確認同一個款式會依體型長出不同大小的網格。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\design_to_mesh.py

環境變數：
    GARMENTCODE_ROOT  GarmentCodeRC 原始碼位置，預設 third_party/GarmentCodeRC
    GC_OUTPUT         輸出位置，預設 output/chatgarment
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
GC_ROOT = Path(os.environ.get("GARMENTCODE_ROOT", REPO / "third_party" / "GarmentCodeRC"))
OUT_ROOT = Path(os.environ.get("GC_OUTPUT", REPO / "output" / "chatgarment")).resolve()
DESIGN = REPO / "third_party" / "ChatGarment" / "example_data" / "example_sewing_patterns" / "example_shirt" / "design.yaml"

# GarmentCode 用相對路徑讀 assets，要在它的根目錄執行
os.chdir(GC_ROOT)
sys.path.insert(0, str(GC_ROOT))

from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from pygarment.meshgen.boxmeshgen import BoxMesh  # noqa: E402
from pygarment.meshgen.sim_config import PathCofig  # noqa: E402
import pygarment.data_config as data_config  # noqa: E402


def ensure_system_json():
    """GarmentCode 需要 system.json 指定路徑；不存在就依範本產生。"""
    path = GC_ROOT / "system.json"
    if path.exists():
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    system = {
        "output": str(OUT_ROOT),
        "datasets_path": str(OUT_ROOT / "datasets"),
        "datasets_sim": str(OUT_ROOT / "datasets_sim"),
        "sim_configs_path": "./assets/Sim_props",
        "bodies_default_path": "./assets/bodies",
        "body_samples_path": "",
    }
    path.write_text(json.dumps(system, indent=2), encoding="utf-8")


def body_variants():
    """測試用的身體：GarmentCode 內建的女性、男性平均體型，以及男性體型只把胸圍加 12 cm。
    正式版改由紀泓宇的 SMPL-X 本人模型量測。"""
    bodies_dir = GC_ROOT / "assets" / "bodies"
    out = {}
    for name in ["mean_female", "mean_male"]:
        out[name] = bodies_dir / f"{name}.yaml"

    big = yaml.safe_load((bodies_dir / "mean_male.yaml").read_text())
    big["body"]["bust"] += 12
    big["body"]["underbust"] += 12
    path = OUT_ROOT / "bodies" / "mean_male_bust+12.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(big), encoding="utf-8")
    out["mean_male_bust+12"] = path
    return out


def generate(design, body_file, name, resolution=None):
    """款式 + 身體 → 2D 版片 → 3D 初始網格（已縫合、帶 UV，未垂墜）。
    resolution：頂點間距（cm），預設用 GarmentCode 的 1.0；頂點數大約和它的平方成反比。"""
    body = BodyParameters(str(body_file))
    garment = MetaGarment(name, body, design)
    pattern = garment.assembly()
    if garment.is_self_intersecting():
        print(f"  [警告] {name} 版片自我相交")

    folder = Path(pattern.serialize(
        OUT_ROOT, tag="", to_subfolder=True,
        with_3d=False, with_text=False, view_ids=False, with_printable=False,
    ))
    body.save(folder)

    spec = next(folder.glob("*_specification.json"))
    props = data_config.Properties("./assets/Sim_props/default_sim_props.yaml")
    paths = PathCofig(
        in_element_path=folder, out_path=OUT_ROOT,
        in_name=spec.stem.rpartition("_")[0],
        body_name="mean_all", smpl_body=False, add_timestamp=False,
    )
    box = BoxMesh(paths.in_g_spec, resolution or props["sim"]["config"]["resolution_scale"])
    box.load()
    box.serialize(paths, store_panels=False, uv_config=props["render"]["config"]["uv_texture"])
    return body, box, paths


def pattern_girth(spec_file):
    """衣服軀幹的周長（cm）：把 2D 版片中所有軀幹版片（前、後片）的寬度加總。
    不從 3D box mesh 橫切量，因為那時版片還沒縫合、彼此有間隙，切出來的數字不可靠。"""
    panels = json.loads(Path(spec_file).read_text())["pattern"]["panels"]
    total = 0.0
    for name, panel in panels.items():
        if "torso" in name:
            xs = [v[0] for v in panel["vertices"]]
            total += max(xs) - min(xs)
    return total


def main():
    ensure_system_json()
    design = yaml.safe_load(DESIGN.read_text())["design"]
    print(f"款式：{DESIGN.relative_to(REPO)}")

    rows = []
    for name, body_file in body_variants().items():
        print(f"\n[{name}]")
        body, box, paths = generate(design, body_file, f"shirt_{name}")
        v = np.asarray(box.vertices)
        f = np.asarray(box.faces)
        p = body.params
        girth = pattern_girth(paths.in_g_spec)
        print(f"  網格：{paths.g_box_mesh}")
        print(f"  頂點 {len(v)}，三角面 {len(f)}，版片 {len(box.panels)} 片")
        rows.append((name, p["bust"], girth, v[:, 1].max() - v[:, 1].min()))

    print("\n| 身體 | 身體胸圍 | 衣服軀幹周長（版片寬度加總） | 餘量 | 衣服高度 |")
    print("|---|---|---|---|---|")
    for name, bust, girth, h in rows:
        print(f"| {name} | {bust:.1f} | {girth:.1f} | {girth - bust:+.1f} | {h:.1f} |")


if __name__ == "__main__":
    main()
