# -*- coding: utf-8 -*-
r"""試驗：會收緊的領圈（像 T 恤的羅紋領）。
GarmentCode 的 Turtle 布條跟領口一樣長，縫上去是直直立起來的矮立領（見 neckband_tee.py）。
這裡另外定義 RibBand：
  - 布條長度 ＝ 領口長度 × RIB_RATIO（0.85），縫上去時被拉開，像真的羅紋領
  - 縫合時只有布條（版片名稱含 collar_）的邊拉長限制放寬到 RIB_STRETCH（羅紋有彈性），其他布照舊 2%
  - 布條上緣沒有東西拉，會縮回原長 → 布條往內收、貼著脖子
其餘同 neckband_tee.py 的 band：領口加寬 5 cm、前領深 9.5＋2 cm、後領深多挖 2 cm、寬 2 cm。不改 GarmentCode 原始碼。

跑法（在 repo 根目錄；這台筆電要從英文路徑副本跑）：
    .venv\Scripts\python flows\body_fit\experiments\neckband_rib.py [收緊比例，預設 0.85] [AI 結果資料夾，預設 i01]
輸出：output/physics_export/neckband_tee/rib<比例>/<衣服>/
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import export_neutral as en  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
import neckline  # noqa: E402
import pygarment as pyg  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs import collars  # noqa: E402
from assets.garment_programs.bands import StraightBandPanel  # noqa: E402
import igl  # noqa: E402
import smplx_body  # noqa: E402
import sleevelen  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions  # noqa: E402

WIDEN_CM = 5.0
FRONT_DEPTH_CM = 9.5
BAND_CM = 2.0
RIB_STRETCH = 0.35
RIB_RATIO = 0.85
DEFAULT = en.dm.REPO / "results" / "chatgarment_ai" / "bf16_i" / "i01_blue_crew_tee"


class RibBand(pyg.Component):
    """照 collars.Turtle 寫，只差布條長度乘上 RIB_RATIO。"""

    def __init__(self, tag, body, design) -> None:
        super().__init__(f"RibBand_{tag}")
        depth = design["collar"]["component"]["depth"]["v"]
        f_collar = collars.CircleNeckHalf(design["collar"]["fc_depth"]["v"], design["collar"]["width"]["v"])
        b_collar = collars.CircleNeckHalf(design["collar"]["bc_depth"]["v"], design["collar"]["width"]["v"])
        self.interfaces = {"front_proj": pyg.Interface(self, f_collar), "back_proj": pyg.Interface(self, b_collar)}
        length_f, length_b = f_collar.length() * RIB_RATIO, b_collar.length() * RIB_RATIO
        height_p = body["height"] - body["head_l"] + depth
        self.front = StraightBandPanel(f"{tag}_collar_front", length_f, depth).translate_by([-length_f / 2, height_p, 10])
        self.back = StraightBandPanel(f"{tag}_collar_back", length_b, depth).translate_by([-length_b / 2, height_p, -10])
        self.stitching_rules.append((self.front.interfaces["right"], self.back.interfaces["right"]))
        self.interfaces.update({
            "front": self.front.interfaces["left"],
            "back": self.back.interfaces["left"],
            "bottom": pyg.Interface.from_multiple(self.front.interfaces["bottom"], self.back.interfaces["bottom"]),
        })

    def length(self):
        return self.interfaces["back"].edges.length()


collars.RibBand = RibBand  # MetaGarment 用 getattr(collars, style) 找領子配件


def build_rib(design, name, body_yaml, body_mesh):
    """同 export_neutral.build，只是布條的邊拉長限制放寬。"""
    _, _, paths = en.dm.generate(design, body_yaml, name, resolution=en.RESOLUTION)
    v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
    edges = mesh_edges(f)
    rest = rest_lengths(v, edges, pickle.loads(paths.g_orig_edge_len.read_bytes()))
    folder = paths.g_box_mesh.parent
    seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
    band = np.array(["collar_" in s for s in seg])
    limit = np.where(band[edges].any(1), RIB_STRETCH, en.SEW["stretch_limit"])
    x = sew(v, edges, rest, body=body_mesh, **dict(en.SEW, stretch_limit=limit))
    write_obj_with_new_positions(paths.g_box_mesh, paths.g_box_mesh.with_name(f"{name}_sewn.obj"), x)
    return folder, x, f, v, sleevelen.sleeve_mask(seg)


def main(ratio=None, src=None):
    global RIB_RATIO
    if ratio:
        RIB_RATIO = float(ratio)
    src = Path(src) if src else DEFAULT
    variant = f"rib{RIB_RATIO:.2f}"
    out = en.dm.REPO / "output" / "physics_export" / "neckband_tee" / variant
    work = out / "_src" / f"{src.name}_{variant}"
    if not work.exists():
        shutil.copytree(src, work)
    en.OUT = out
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    body = BodyParameters(str(body_yaml))
    lo, hi = body["neck_w"], body["_base_sleeve_balance"] - 2
    neckline.FRONT_MIN_RATIO = (FRONT_DEPTH_CM + BAND_CM) / body["neck_w"]
    neckline.BACK_MIN_RATIO += BAND_CM / body["neck_w"]
    load = en.load_design

    def load_rib(result_dir):
        design = load(result_dir)
        p = design["collar"]["width"]["v"]
        design["collar"]["width"]["v"] = (lo + p * (hi - lo) + WIDEN_CM - lo) / (hi - lo)
        design["collar"]["component"]["style"]["v"] = "RibBand"
        design["collar"]["component"]["depth"]["v"] = BAND_CM
        return design
    en.load_design = load_rib
    en.build = build_rib
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    v0, f0, weights = smplx_body.make_body("neutral")
    en.export_one(work, body, body_yaml, (bv * 100, bf), (v0, f0, weights), lift, sleevelen.arms(lift))


if __name__ == "__main__":
    main(*sys.argv[1:])
