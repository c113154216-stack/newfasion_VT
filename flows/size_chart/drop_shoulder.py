# -*- coding: utf-8 -*-
r"""落肩上衣版型 DropShoulderShirt（2026-11-25 新增；GarmentCode 原本的檔案一行都沒改）。

為什麼要新版型：GarmentCode 的 Shirt 是「衣身版片右上角挖掉一塊袖籠」，肩點只能在版片寬度以內，
所以肩寬最多到差不多胸寬（z04 POLO 尺寸表肩 60 > 胸 58，版片只推得到 57）。
真的落肩衣（照片上肩線落到上臂）是這樣打的：袖籠是一條從腋下斜到肩點的直線，肩點可以比腋下更外面。

這裡的衣身半片（前、後一樣，側縫在正側邊）：
    下襬中心 (0,0) → 下襬側邊 (-b,0) → 腋下 A（跟 Shirt 同一條側縫線、同一個高度）
    → 肩點 S（延長的肩線上，x＝打版用肩寬/2−1，和 Shirt 的肩點同一個公式）→ 領口中心
A→S 就是袖籠。袖子的接口是一條同樣長的直線（落肩的袖山是平的），袖山不吃縫。
領子、袖口、下襬羅紋都沿用 GarmentCode 原本的程式。

用法：import 這個檔案就會把 DropShoulderShirt 註冊進 MetaGarment 找得到的地方，design 的 meta.upper 設成 "DropShoulderShirt"。
"""
from copy import deepcopy

import numpy as np
import pygarment as pyg
from scipy.spatial.transform import Rotation as R

from assets.garment_programs import bands, meta_garment, sleeves
from assets.garment_programs.base_classes import BaseBodicePanel
from assets.garment_programs.bodice import BodiceHalf, Shirt

import os

SIDE_FRAC = 0.25  # 每片半身佔整件胸圍的比例：前後一樣寬（側縫在正側邊），落肩衣通常這樣打
ARMHOLE_DEPTH_SCALE = float(os.environ.get("DROP_ARMHOLE_SCALE", "1.0"))  # 袖籠深（腋下多低）× 這個倍數（實驗中）


class DropTorsoHalf(BaseBodicePanel):
    """落肩上衣的前或後半片。"""

    def __init__(self, name, body, design, front):
        super().__init__(name, body, design)
        d = design["shirt"]
        m_width = d["width"]["v"] * body["bust"]
        self.width = w = SIDE_FRAC * m_width
        b = SIDE_FRAC * d["flare"]["v"] * m_width
        tan = np.tan(np.deg2rad(body["_shoulder_incl"]))
        length = d["length"]["v"] * body["waist_line"]  # 前後一樣寬，Shirt 的前片長度修正（fb_diff）是 0
        top = length + tan * w
        sx = body["_base_sleeve_balance"] / 2  # 肩點離中心多遠（Shirt 同一個公式）
        cw = design["sleeve"]["connecting_width"]["v"]  # 袖籠深（BodiceHalf.eval_dep_params 算好的）
        ay = length - cw
        ax = b + (w - b) * ay / length  # 腋下在 Shirt 那條側縫線上
        sy = length + tan * (w - sx)  # 肩線延長到肩點
        self.edges = pyg.EdgeSeqFactory.from_verts([0, 0], [-b, 0], [-ax, ay], [-sx, sy], [0, top], loop=True)
        bottom_ruffle = w / ((body["waist"] - body["waist_back_width"]) / 2) if front else w / (body["waist_back_width"] / 2)
        self.interfaces = {
            "outside": pyg.Interface(self, self.edges[1]),
            "armhole": pyg.Interface(self, self.edges[2]),
            "shoulder": pyg.Interface(self, self.edges[3]),
            "inside": pyg.Interface(self, self.edges[4]),
            "bottom": pyg.Interface(self, self.edges[0], ruffle=bottom_ruffle),
            "shoulder_corner": pyg.Interface(self, [self.edges[2], self.edges[3]]),
            "collar_corner": pyg.Interface(self, [self.edges[3], self.edges[4]]),
        }
        self.translate_by([0, body["height"] - body["head_l"] - top, 0])

    def get_width(self, level):
        """跟 tee.py 一樣（領子用 get_width(0) 算領深修正，回傳原本的版片寬，領口才會和 Shirt 一樣）。"""
        return super().get_width(level) + self.width - self.body["shoulder_w"] / 2


class DropSleeve(pyg.Component):
    """接口是一條直線的袖子（其他和 sleeves.Sleeve 一樣：袖片、袖口、擺放）。arm_len：袖籠那條直線的長度。"""

    def __init__(self, tag, body, design, arm_len):
        super().__init__(f"Sleeve_{tag}")
        design = design["sleeve"]
        self.design = design
        self.body = body
        rest_angle = max(np.deg2rad(design["sleeve_angle"]["v"]), np.deg2rad(body["_shoulder_incl"]))
        cuff_len_adj = sleeves.Sleeve._cuff_len_adj(self)

        def opening():
            # 由下往上的一條直線，傾斜 rest_angle（跟 ArmholeCurve 反轉後的接口同方向）
            return pyg.EdgeSequence(pyg.Edge([0, 0], [arm_len * np.sin(rest_angle), arm_len * np.cos(rest_angle)]))

        self.f_sleeve = sleeves.SleevePanel(f"{tag}_sleeve_f", body, design, opening(),
                                            length_shift=-cuff_len_adj).translate_by([0, 0, 15])
        self.b_sleeve = sleeves.SleevePanel(f"{tag}_sleeve_b", body, design, opening(),
                                            length_shift=-cuff_len_adj).translate_by([0, 0, -15])
        self.stitching_rules = pyg.Stitches(
            (self.f_sleeve.interfaces["top"], self.b_sleeve.interfaces["top"]),
            (self.f_sleeve.interfaces["bottom"], self.b_sleeve.interfaces["bottom"]),
        )
        self.interfaces = {
            "in": pyg.Interface.from_multiple(self.f_sleeve.interfaces["in"],
                                              self.b_sleeve.interfaces["in"].reverse(with_edge_dir_reverse=True)),
            "out": pyg.Interface.from_multiple(self.f_sleeve.interfaces["out"], self.b_sleeve.interfaces["out"]),
        }
        if design["cuff"]["type"]["v"]:  # 以下照抄 sleeves.Sleeve
            cdesign = deepcopy(design)
            cuff_circ = max(self.interfaces["out"].edges.length() / design["cuff"]["top_ruffle"]["v"], body["wrist"])
            cdesign["cuff"]["b_width"] = dict(v=cuff_circ)
            cdesign["cuff"]["cuff_len"]["v"] = cuff_len_adj
            self.cuff = getattr(bands, cdesign["cuff"]["type"]["v"])(f"sl_{tag}", cdesign)
            self.cuff.rotate_by(R.from_euler("XYZ", [0, 0, -90], degrees=True))
            self.cuff.place_by_interface(self.cuff.interfaces["top"], self.interfaces["out"], gap=2, alignment="top")
            self.stitching_rules.append((self.cuff.interfaces["top"], self.interfaces["out"]))
            self.interfaces["out"] = self.cuff.interfaces["bottom"]
        self.rotate_by(R.from_euler("XYZ", [0, 0, body["arm_pose_angle"]], degrees=True))
        self.set_panel_label("arm")

    def length(self):
        if self.design["cuff"]["type"]["v"]:
            return self.f_sleeve.length() + self.cuff.length()
        return self.f_sleeve.length()


class DropShoulderHalf(BodiceHalf):
    """BodiceHalf 換成落肩衣身和直線袖籠的袖子；領子、縫合規則照舊。"""

    def __init__(self, name, body, design, fitted=False):
        pyg.Component.__init__(self, name)
        design = deepcopy(design)
        self.sleeve = None
        self.collar_comp = None
        # 袖籠深：照 BodiceHalf.eval_dep_params 的公式（它要用到版片，所以先算好再做版片）
        v = design["sleeve"]["connecting_width"]["v"]
        design["sleeve"]["connecting_width"]["v"] = body["_armscye_depth"] * (1 + v) * ARMHOLE_DEPTH_SCALE
        self.ftorso = DropTorsoHalf(f"{name}_ftorso", body, design, front=True).translate_by([0, 0, 30])
        self.btorso = DropTorsoHalf(f"{name}_btorso", body, design, front=False).translate_by([0, 0, -25])
        self.interfaces.update({
            "f_bottom": self.ftorso.interfaces["bottom"], "b_bottom": self.btorso.interfaces["bottom"],
            "front_in": self.ftorso.interfaces["inside"], "back_in": self.btorso.interfaces["inside"],
        })
        cw = design["sleeve"]["connecting_width"]["v"]
        self._collar_params(body, design)
        design["sleeve"]["connecting_width"]["v"] = cw  # eval_dep_params 也會改袖籠深，換回我們的
        self.add_sleeves(name, body, design)
        self.add_collars(name, body, design)
        self.stitching_rules.append((self.ftorso.interfaces["shoulder"], self.btorso.interfaces["shoulder"]))
        self.stitching_rules.append((self.ftorso.interfaces["outside"], self.btorso.interfaces["outside"]))

    def _collar_params(self, body, design):
        BodiceHalf.eval_dep_params(self, body, design)

    def add_sleeves(self, name, body, design):
        if design["sleeve"]["sleeveless"]["v"]:
            return
        f_arm, b_arm = self.ftorso.interfaces["armhole"], self.btorso.interfaces["armhole"]
        self.sleeve = DropSleeve(name, body, design, f_arm.edges.length())
        # 袖籠介面的順序和 BodiceHalf.add_sleeves（cut_corner 插進去的那段）一樣
        bodice_int = pyg.Interface.from_multiple(f_arm.reverse(with_edge_dir_reverse=True), b_arm.reverse())
        self.stitching_rules.append((self.sleeve.interfaces["in"], bodice_int))
        self.sleeve.place_by_interface(self.sleeve.interfaces["in"], bodice_int,
                                       gap=-1 - body["arm_pose_angle"] / 10, alignment="top")
        f_arm.edges.propagate_label(f"{self.name}_armhole")
        b_arm.edges.propagate_label(f"{self.name}_armhole")


class DropShoulderShirt(Shirt):
    """落肩上衣（meta.upper＝"DropShoulderShirt"）。"""

    def __init__(self, body, design, fitted=False):
        pyg.Component.__init__(self, self.__class__.__name__)
        design = self.eval_dep_params(design)
        self.right = DropShoulderHalf("right", body, design)
        self.left = DropShoulderHalf("left", body, design["left"] if design["left"]["enable_asym"]["v"] else design).mirror()
        if ("openfront" not in design["shirt"]) or (not design["shirt"]["openfront"]["v"]):
            self.stitching_rules.append((self.right.interfaces["front_in"], self.left.interfaces["front_in"]))
        self.stitching_rules.append((self.right.interfaces["back_in"], self.left.interfaces["back_in"]))
        self.interfaces = {"bottom": pyg.Interface.from_multiple(
            self.right.interfaces["f_bottom"].reverse(), self.left.interfaces["f_bottom"],
            self.left.interfaces["b_bottom"].reverse(), self.right.interfaces["b_bottom"])}


meta_garment.DropShoulderShirt = DropShoulderShirt  # MetaGarment 用 globals()[名稱] 找上衣類別
