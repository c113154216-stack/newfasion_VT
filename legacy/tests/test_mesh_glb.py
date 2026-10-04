"""
adapters/mesh_glb.py 工具函式的單元測試。

測試重點：
  - export_glb / load_glb 完整 round-trip（頂點、面數量與數值一致）
  - save_sidecar / load_sidecar 完整 round-trip（任意 dict）
  - load_sidecar 在沒有 sidecar 檔案時回傳空 dict
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def simple_mesh():
    """最小的三角錐（4 頂點 4 面），方便驗算 round-trip 是否精確。"""
    vertices = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2],
         [0, 1, 3],
         [0, 2, 3],
         [1, 2, 3]],
        dtype=np.int64,
    )
    return vertices, faces


# --------------------------------------------------------------------------- #
# GLB 匯出 / 讀入 round-trip
# --------------------------------------------------------------------------- #

class TestExportLoadGlb:
    def test_output_file_created(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_roundtrip_vertex_count(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb, load_glb

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)
        v2, f2 = load_glb(out)
        assert v2.shape[0] == v.shape[0]

    def test_roundtrip_face_count(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb, load_glb

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)
        v2, f2 = load_glb(out)
        assert f2.shape[0] == f.shape[0]

    def test_roundtrip_vertex_values(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb, load_glb

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)
        v2, _ = load_glb(out)
        # 頂點順序可能被 trimesh 重排，用 sorted rows 比較
        v_sorted = np.array(sorted(v.tolist()))
        v2_sorted = np.array(sorted(v2.tolist()))
        np.testing.assert_allclose(v_sorted, v2_sorted, atol=1e-5)

    def test_parent_dir_auto_created(self, simple_mesh, tmp_path):
        """export_glb 應自動建立不存在的父目錄。"""
        from adapters.mesh_glb import export_glb

        v, f = simple_mesh
        out = tmp_path / "nested" / "dir" / "mesh.glb"
        export_glb(v, f, out)
        assert out.exists()

    def test_load_returns_numpy(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb, load_glb

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)
        v2, f2 = load_glb(out)
        assert isinstance(v2, np.ndarray)
        assert isinstance(f2, np.ndarray)


# --------------------------------------------------------------------------- #
# Sidecar JSON round-trip
# --------------------------------------------------------------------------- #

class TestSidecar:
    def test_basic_roundtrip(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb, load_sidecar, save_sidecar

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)

        data = {"betas": [0.1, -0.2, 0.3], "gender": "neutral"}
        save_sidecar(out, data)
        loaded = load_sidecar(out)

        assert loaded["gender"] == "neutral"
        assert loaded["betas"] == pytest.approx([0.1, -0.2, 0.3])

    def test_missing_sidecar_returns_empty_dict(self, tmp_path):
        """找不到 sidecar 時應回傳空 dict，不應 raise 任何例外。"""
        from adapters.mesh_glb import load_sidecar

        result = load_sidecar(tmp_path / "nonexistent.glb")
        assert result == {}

    def test_sidecar_file_path(self, simple_mesh, tmp_path):
        """sidecar 存在 <glb_path>.json。"""
        from adapters.mesh_glb import export_glb, save_sidecar

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)
        save_sidecar(out, {"test": True})
        sidecar = out.parent / (out.name + ".json")
        assert sidecar.exists()

    def test_nested_dict_roundtrip(self, simple_mesh, tmp_path):
        from adapters.mesh_glb import export_glb, load_sidecar, save_sidecar

        v, f = simple_mesh
        out = tmp_path / "mesh.glb"
        export_glb(v, f, out)

        payload = {"has_sleeve_topology": True, "meta": {"version": 2}}
        save_sidecar(out, payload)
        assert load_sidecar(out) == payload
