# -*- coding: utf-8 -*-
"""
用 GarmentCode（pygarment）生成 T-shirt：設計參數 → 2D 版片 → 3D 初始網格（box mesh，含 UV）。

不跑布料模擬（需要 NVIDIA Warp 分支，本機沒有顯卡），所以 3D 網格是「版片圍在身體外、
已縫合但還沒垂墜」的狀態。

跑法（在 D 槽的 venv 裡）：
    D:\\newfasion_VT_work\\.venv\\Scripts\\python tools\\garmentcode\\gen_tshirt.py

環境變數：
    GARMENTCODE_ROOT  GarmentCode 原始碼位置，預設 D:\\newfasion_VT_work\\GarmentCode
    GC_OUTPUT         輸出位置，預設 D:\\newfasion_VT_work\\output\\garmentcode
"""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

GC_ROOT = Path(os.environ.get("GARMENTCODE_ROOT", r"D:\newfasion_VT_work\GarmentCode"))
OUT_ROOT = Path(os.environ.get("GC_OUTPUT", r"D:\newfasion_VT_work\output\garmentcode"))

# GarmentCode 的程式用相對路徑讀 assets，要在它的根目錄執行
os.chdir(GC_ROOT)
sys.path.insert(0, str(GC_ROOT))

from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from pygarment.meshgen.boxmeshgen import BoxMesh  # noqa: E402
from pygarment.meshgen.sim_config import PathCofig  # noqa: E402
import pygarment.data_config as data_config  # noqa: E402


def ensure_system_json():
    """GarmentCode 需要 system.json 指定各種路徑；不存在就依範本產生，輸出指到 D 槽。"""
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


def generate_pattern(design_file, body_file, name):
    """設計參數 + 人體尺寸 → 2D 版片（*_specification.json、版片圖）。回傳輸出資料夾。"""
    body = BodyParameters(body_file)
    with open(design_file, "r") as f:
        design = yaml.safe_load(f)["design"]

    garment = MetaGarment(name, body, design)
    pattern = garment.assembly()
    if garment.is_self_intersecting():
        print(f"[警告] {name} 版片自我相交")

    folder = pattern.serialize(
        OUT_ROOT,
        tag="_" + datetime.now().strftime("%y%m%d-%H-%M-%S"),
        to_subfolder=True,
        with_3d=False, with_text=False, view_ids=False,
        with_printable=True,
    )
    body.save(folder)
    shutil.copy(design_file, Path(folder) / "design_params.yaml")
    return Path(folder)


def generate_box_mesh(pattern_folder, body_name="mean_all"):
    """2D 版片 → 3D 初始網格（縫合、帶 UV），不模擬。"""
    spec = next(pattern_folder.glob("*_specification.json"))
    garment_name = spec.stem.rpartition("_")[0]

    props = data_config.Properties("./assets/Sim_props/default_sim_props.yaml")
    paths = PathCofig(
        in_element_path=pattern_folder,
        out_path=OUT_ROOT,
        in_name=garment_name,
        body_name=body_name,
        smpl_body=False,
        add_timestamp=True,
    )

    box = BoxMesh(paths.in_g_spec, props["sim"]["config"]["resolution_scale"])
    box.load()
    box.serialize(paths, store_panels=False, uv_config=props["render"]["config"]["uv_texture"])
    return paths, box


def main():
    ensure_system_json()

    pattern_folder = generate_pattern(
        design_file="./assets/design_params/t-shirt.yaml",
        body_file="./assets/bodies/mean_all.yaml",
        name="tshirt",
    )
    print(f"2D 版片：{pattern_folder}")

    paths, box = generate_box_mesh(pattern_folder)
    print(f"3D 網格：{paths.g_box_mesh}")
    print(f"  頂點數 {len(box.vertices)}，三角面數 {len(box.faces)}，版片 {list(box.panels.keys())}")


if __name__ == "__main__":
    main()
