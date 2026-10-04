"""
geometry_physics/body_model.py 的單元測試。

因為正式的 SMPL-X 授權模型不一定存在（`.npz` 需手動下載），
本測試只覆蓋「不需要授權模型」的純幾何部分：

  - build_proxy_mannequin：幾何替代人偶（永遠可以執行）
  - smplx_body_from_measurements：當 SMPL-X 權重存在時才執行；
    沒有時用 pytest.mark.skipif 跳過而非讓整個測試套件失敗。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_GEO_DIR = Path(__file__).parent.parent / "geometry_physics"
if str(_GEO_DIR) not in sys.path:
    sys.path.insert(0, str(_GEO_DIR))

_SMPLX_DIR = _GEO_DIR / "assets" / "smplx"
_HAS_SMPLX_WEIGHTS = (
    (_SMPLX_DIR / "SMPLX_NEUTRAL.npz").exists()
    or (_SMPLX_DIR / "SMPLX_MALE.npz").exists()
    or (_SMPLX_DIR / "SMPLX_FEMALE.npz").exists()
)

try:
    import smplx as _smplx_pkg
    import torch as _torch_pkg
    _HAS_SMPLX_PKG = True
except ImportError:
    _HAS_SMPLX_PKG = False

_CAN_RUN_SMPLX = _HAS_SMPLX_PKG and _HAS_SMPLX_WEIGHTS

needs_smplx = pytest.mark.skipif(
    not _CAN_RUN_SMPLX,
    reason="需要 smplx/torch 套件 + SMPLX_*.npz 授權模型",
)


# --------------------------------------------------------------------------- #
# build_proxy_mannequin
# --------------------------------------------------------------------------- #

class TestBuildProxyMannequin:
    def test_returns_tuple_of_two(self):
        import body_model as bm

        result = bm.build_proxy_mannequin()
        assert len(result) == 2

    def test_vertex_shape(self):
        import body_model as bm

        v, _ = bm.build_proxy_mannequin()
        assert v.ndim == 2
        assert v.shape[1] == 3, "頂點應為 (N, 3)"

    def test_face_shape(self):
        import body_model as bm

        _, f = bm.build_proxy_mannequin()
        assert f.ndim == 2
        assert f.shape[1] == 3, "面索引應為 (M, 3)"

    def test_vertex_count_reasonable(self):
        import body_model as bm

        v, _ = bm.build_proxy_mannequin()
        # radial_segments=32 * height_segments=48 → 約 1600+ 頂點
        assert v.shape[0] > 500

    def test_face_indices_in_range(self):
        import body_model as bm

        v, f = bm.build_proxy_mannequin()
        assert f.min() >= 0
        assert f.max() < v.shape[0]

    def test_height_scales_y_axis(self):
        """身高較高的 proxy 人偶，Y 軸最大值應等比增加。"""
        import body_model as bm

        v_170, _ = bm.build_proxy_mannequin(height_cm=170.0)
        v_180, _ = bm.build_proxy_mannequin(height_cm=180.0)
        assert v_180[:, 1].max() > v_170[:, 1].max()

    def test_y_range_matches_height(self):
        """Y 軸最大值應接近 height_cm 換算成公尺。"""
        import body_model as bm

        h = 175.0
        v, _ = bm.build_proxy_mannequin(height_cm=h)
        y_max = v[:, 1].max()
        assert pytest.approx(y_max, abs=0.05) == h / 100.0

    def test_wider_bust_wider_body(self):
        """胸圍較大的人偶，X/Z 軸延伸應更寬。"""
        import body_model as bm

        v_slim, _ = bm.build_proxy_mannequin(bust_cm=80.0)
        v_wide, _ = bm.build_proxy_mannequin(bust_cm=110.0)
        assert v_wide[:, 0].max() > v_slim[:, 0].max()

    def test_default_params_run(self):
        """無參數呼叫不應 raise。"""
        import body_model as bm

        v, f = bm.build_proxy_mannequin()
        assert v is not None

    def test_custom_segments(self):
        """較少的 segment 應產生較少頂點。"""
        import body_model as bm

        v_low, _ = bm.build_proxy_mannequin(radial_segments=8, height_segments=8)
        v_high, _ = bm.build_proxy_mannequin(radial_segments=32, height_segments=32)
        assert v_low.shape[0] < v_high.shape[0]


# --------------------------------------------------------------------------- #
# smplx_body_from_measurements（需要 SMPL-X 套件 + 授權模型）
# --------------------------------------------------------------------------- #

class TestSmplxBodyFromMeasurements:
    @needs_smplx
    def test_returns_three_values(self):
        import body_model as bm

        result = bm.smplx_body_from_measurements(
            height_cm=170, bust_cm=90, waist_cm=75, hip_cm=95
        )
        assert len(result) == 3, "應回傳 (vertices, faces, betas)"

    @needs_smplx
    def test_vertices_shape(self):
        import body_model as bm

        v, f, betas = bm.smplx_body_from_measurements(
            height_cm=170, bust_cm=90, waist_cm=75, hip_cm=95
        )
        assert v.ndim == 2 and v.shape[1] == 3

    @needs_smplx
    def test_faces_triangulated(self):
        import body_model as bm

        _, f, _ = bm.smplx_body_from_measurements(
            height_cm=170, bust_cm=90, waist_cm=75, hip_cm=95
        )
        assert f.shape[1] == 3, "面索引應為三角形"

    @needs_smplx
    def test_betas_is_numpy_array(self):
        import body_model as bm

        _, _, betas = bm.smplx_body_from_measurements(
            height_cm=170, bust_cm=90, waist_cm=75, hip_cm=95
        )
        assert isinstance(betas, np.ndarray)

    @needs_smplx
    def test_height_affects_y_range(self):
        """不同身高的人體，Y 軸範圍應有明顯差異。"""
        import body_model as bm

        v_170, _, _ = bm.smplx_body_from_measurements(170, 90, 75, 95)
        v_185, _, _ = bm.smplx_body_from_measurements(185, 90, 75, 95)
        y_range_170 = v_170[:, 1].max() - v_170[:, 1].min()
        y_range_185 = v_185[:, 1].max() - v_185[:, 1].min()
        assert y_range_185 > y_range_170

    @needs_smplx
    def test_gender_neutral_runs(self):
        import body_model as bm

        v, f, _ = bm.smplx_body_from_measurements(
            170, 90, 75, 95, gender="neutral"
        )
        assert v.shape[0] > 0
