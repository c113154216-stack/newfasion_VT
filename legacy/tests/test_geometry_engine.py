"""
geometry_physics/geometry_engine.py 演算法的單元測試。

測試重點：
  - load_obj_vf：正確解析現有的 .obj 公版檔案
  - build_real_shirt_template：回傳 (vertices, weights, faces)，形狀符合預期
  - deform_vertices：identity 參數不改變頂點；縮放參數確實改變頂點；空陣列安全
  - export_obj：輸出合法的 OBJ 格式
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 確保 geometry_physics 在 import 路徑中
_GEO_DIR = Path(__file__).parent.parent / "geometry_physics"
if str(_GEO_DIR) not in sys.path:
    sys.path.insert(0, str(_GEO_DIR))

GARMENT_DIR = _GEO_DIR / "assets" / "garments"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _garment_obj(name: str) -> Path:
    """回傳公版服飾 .obj 路徑；找不到就 skip。"""
    p = GARMENT_DIR / name
    if not p.exists():
        pytest.skip(f"公版服飾檔案不存在：{p}")
    return p


# --------------------------------------------------------------------------- #
# load_obj_vf
# --------------------------------------------------------------------------- #

class TestLoadObjVf:
    def test_tshirt_vertex_count(self):
        import geometry_engine as ge

        v, f = ge.load_obj_vf(str(_garment_obj("tshirt_base.obj")))
        assert v.shape[1] == 3, "頂點應為 (N, 3)"
        assert v.shape[0] > 100, "公版上衣頂點數應 > 100"

    def test_tshirt_face_count(self):
        import geometry_engine as ge

        v, f = ge.load_obj_vf(str(_garment_obj("tshirt_base.obj")))
        assert f.shape[1] == 3, "面索引應為 (M, 3) 三角形"
        assert f.shape[0] > 0, "面數量應 > 0"

    def test_pants_vertex_count(self):
        import geometry_engine as ge

        v, f = ge.load_obj_vf(str(_garment_obj("pants_base.obj")))
        assert v.shape[0] > 100

    def test_face_indices_in_range(self):
        """所有面索引都必須落在 [0, n_vertices) 範圍內。"""
        import geometry_engine as ge

        v, f = ge.load_obj_vf(str(_garment_obj("tshirt_base.obj")))
        assert f.min() >= 0
        assert f.max() < v.shape[0]

    def test_returns_numpy_float64(self):
        import geometry_engine as ge

        v, f = ge.load_obj_vf(str(_garment_obj("tshirt_base.obj")))
        assert v.dtype == np.float64
        assert f.dtype == np.int64

    def test_minimal_obj_string(self, tmp_path):
        """用寫入暫存 .obj 測試解析器，不依賴公版資產。"""
        import geometry_engine as ge

        obj_text = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "v 0.0 0.0 1.0\n"
            "f 1 2 3\n"
            "f 1 2 4\n"
        )
        p = tmp_path / "test.obj"
        p.write_text(obj_text, encoding="utf-8")
        v, f = ge.load_obj_vf(str(p))
        assert v.shape == (4, 3)
        assert f.shape == (2, 3)

    def test_quad_face_splits_to_two_triangles(self, tmp_path):
        """四邊形面應被切成兩個三角形。"""
        import geometry_engine as ge

        obj_text = (
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "f 1 2 3 4\n"
        )
        p = tmp_path / "quad.obj"
        p.write_text(obj_text, encoding="utf-8")
        _, f = ge.load_obj_vf(str(p))
        assert f.shape == (2, 3), "一個四邊形應拆成 2 個三角形"


# --------------------------------------------------------------------------- #
# build_real_shirt_template
# --------------------------------------------------------------------------- #

class TestBuildRealShirtTemplate:
    def test_returns_three_values(self):
        import geometry_engine as ge

        result = ge.build_real_shirt_template(str(_garment_obj("tshirt_base.obj")))
        assert len(result) == 3, "應回傳 (vertices, weights, faces)"

    def test_weights_sum_to_one(self):
        """每個頂點的 4 個權重總和應 = 1.0。"""
        import geometry_engine as ge

        v, w, f = ge.build_real_shirt_template(str(_garment_obj("tshirt_base.obj")))
        sums = w.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(len(sums)), atol=1e-6)

    def test_weights_shape(self):
        import geometry_engine as ge

        v, w, f = ge.build_real_shirt_template(str(_garment_obj("tshirt_base.obj")))
        assert w.shape == (v.shape[0], 4), "weights 形狀應為 (N, 4)"

    def test_weights_non_negative(self):
        import geometry_engine as ge

        _, w, _ = ge.build_real_shirt_template(str(_garment_obj("tshirt_base.obj")))
        assert (w >= 0).all(), "所有權重值應 >= 0"

    def test_face_indices_in_range(self):
        import geometry_engine as ge

        v, w, f = ge.build_real_shirt_template(str(_garment_obj("tshirt_base.obj")))
        assert f.max() < v.shape[0]


# --------------------------------------------------------------------------- #
# deform_vertices
# --------------------------------------------------------------------------- #

class TestDeformVertices:
    def _make_box_mesh(self):
        """最小方塊網格，用於算法驗算。"""
        import geometry_engine as ge

        v = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
             [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        )
        w = np.ones((8, 4)) / 4.0  # 均等分配
        return v, w

    def test_identity_params_preserves_vertices(self):
        """全部 100 的 params 應回傳與原始頂點幾乎完全相同的結果。"""
        import geometry_engine as ge

        v, w = self._make_box_mesh()
        params = {"length": 100, "shoulder": 100, "chest": 100, "waist": 100}
        out = ge.deform_vertices(v, w, params)
        np.testing.assert_allclose(out, v, atol=1e-10)

    def test_length_scale_y_axis(self):
        """length=200 應把 Y 軸方向放大為 2 倍。"""
        import geometry_engine as ge

        v, w = self._make_box_mesh()
        params = {"length": 200, "shoulder": 100, "chest": 100, "waist": 100}
        out = ge.deform_vertices(v, w, params)
        # Y=1.0 的頂點被拉伸成 Y=2.0
        y_before = v[:, 1]
        y_after = out[:, 1]
        for yb, ya in zip(y_before, y_after):
            assert pytest.approx(ya, abs=1e-9) == yb * 2.0

    def test_output_same_vertex_count(self):
        import geometry_engine as ge

        v, w = self._make_box_mesh()
        out = ge.deform_vertices(v, w, {"length": 120, "chest": 110})
        assert out.shape[0] == v.shape[0]

    def test_output_is_3d(self):
        import geometry_engine as ge

        v, w = self._make_box_mesh()
        out = ge.deform_vertices(v, w, {})
        assert out.shape[1] == 3

    def test_empty_vertices_safe(self):
        """空陣列不應 raise，直接回傳空陣列。"""
        import geometry_engine as ge

        v = np.zeros((0, 3), dtype=np.float64)
        w = np.zeros((0, 4), dtype=np.float64)
        out = ge.deform_vertices(v, w, {"length": 100})
        assert out.shape == (0, 3)

    def test_real_garment_deform(self):
        """對真實公版服飾套用尺寸縮放，確認頂點數量不變、值有改變。"""
        import geometry_engine as ge

        v, w, _ = ge.build_real_shirt_template(str(_garment_obj("tshirt_base.obj")))
        params = {"length": 110, "shoulder": 108, "chest": 105, "waist": 102}
        out = ge.deform_vertices(v, w, params)
        assert out.shape == v.shape
        assert not np.allclose(out, v), "縮放後頂點值應有改變"


# --------------------------------------------------------------------------- #
# export_obj
# --------------------------------------------------------------------------- #

class TestExportObj:
    def test_output_file_created(self, tmp_path):
        import geometry_engine as ge

        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        f = np.array([[0, 1, 2]])
        out = tmp_path / "out.obj"
        ge.export_obj(v, f, str(out))
        assert out.exists()

    def test_output_has_vertex_lines(self, tmp_path):
        import geometry_engine as ge

        v = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        f = np.array([[0, 1, 2]])
        out = tmp_path / "out.obj"
        ge.export_obj(v, f, str(out))
        lines = out.read_text(encoding="utf-8").splitlines()
        v_lines = [l for l in lines if l.startswith("v ")]
        f_lines = [l for l in lines if l.startswith("f ")]
        assert len(v_lines) == 3
        assert len(f_lines) == 1

    def test_face_indices_are_1based(self, tmp_path):
        """OBJ 格式面索引從 1 開始。"""
        import geometry_engine as ge

        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        f = np.array([[0, 1, 2]])
        out = tmp_path / "out.obj"
        ge.export_obj(v, f, str(out))
        f_line = [l for l in out.read_text().splitlines() if l.startswith("f ")][0]
        indices = [int(x) for x in f_line.split()[1:]]
        assert indices == [1, 2, 3]
