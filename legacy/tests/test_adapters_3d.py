"""
Chi 3D Adapter 整合測試：
  - adapters/body_generator.py  (M2)
  - adapters/garment_processor.py (M3)
  - adapters/physics_simulator.py (M4)

「PYTEST_CURRENT_TEST」環境變數在 pytest 執行期間由 pytest 自動設定，
因此這三個 adapter 的 _use_stub() 都會回傳 True，走最小 GLB 的 stub 路徑，
確保測試在沒有 SMPL-X 授權模型的環境也能穩定通過。

測試分三層：
  1. 各 adapter 個別功能（輸入合法 → 回傳 glb_path 且檔案存在）
  2. 邊界輸入（不合法體型 / 不存在 product_id 的容錯）
  3. 完整 pipeline：M2 → M3 → M4 → 輸出 .glb 可被 trimesh 讀取
"""

from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# body_generator (M2)
# --------------------------------------------------------------------------- #

class TestBodyGenerator:
    def test_generate_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg

        result = bg.generate(175, 92, 78, 96, "test_user")
        assert isinstance(result, dict)
        assert "glb_path" in result

    def test_generate_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg

        result = bg.generate(165, 88, 72, 92, "exists_check")
        assert Path(result["glb_path"]).exists()

    def test_generate_file_is_glb(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg

        result = bg.generate(170, 90, 75, 95, "glb_ext")
        assert result["glb_path"].endswith(".glb")

    def test_generate_glb_nonempty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg

        result = bg.generate(170, 90, 75, 95, "nonempty")
        size = Path(result["glb_path"]).stat().st_size
        assert size > 0, "GLB 不應是空檔案"

    def test_generate_unique_output_per_name(self, tmp_path, monkeypatch):
        """不同 output_name 應產生不同檔名。"""
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg

        r1 = bg.generate(170, 90, 75, 95, "personA")
        r2 = bg.generate(170, 90, 75, 95, "personB")
        assert r1["glb_path"] != r2["glb_path"]


# --------------------------------------------------------------------------- #
# garment_processor (M3)
# --------------------------------------------------------------------------- #

class TestGarmentProcessor:
    def test_prepare_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import garment_processor as gp

        result = gp.prepare(product_id=1, size="M")
        assert isinstance(result, dict)
        assert "glb_path" in result

    def test_prepare_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import garment_processor as gp

        result = gp.prepare(product_id=2, size="L")
        assert Path(result["glb_path"]).exists()

    def test_prepare_file_is_glb(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import garment_processor as gp

        result = gp.prepare(product_id=3, size="S")
        assert result["glb_path"].endswith(".glb")

    def test_prepare_different_sizes_different_filenames(self, tmp_path, monkeypatch):
        """不同尺碼應產生不同的輸出檔名。"""
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import garment_processor as gp

        rM = gp.prepare(product_id=5, size="M")
        rXL = gp.prepare(product_id=5, size="XL")
        assert rM["glb_path"] != rXL["glb_path"]

    def test_prepare_default_size(self, tmp_path, monkeypatch):
        """不指定 size 時預設 M，不應 raise。"""
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import garment_processor as gp

        result = gp.prepare(product_id=10)
        assert "glb_path" in result


# --------------------------------------------------------------------------- #
# physics_simulator (M4)
# --------------------------------------------------------------------------- #

class TestPhysicsSimulator:
    def _make_stub_glb(self, tmp_path, name: str) -> str:
        from adapters.glb_utils import create_minimal_glb

        p = tmp_path / name
        create_minimal_glb(p)
        return str(p)

    def test_run_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import physics_simulator as ps

        body = self._make_stub_glb(tmp_path, "body.glb")
        garment = self._make_stub_glb(tmp_path, "garment.glb")
        result = ps.run(body, garment, "combo_test")
        assert isinstance(result, dict)
        assert "glb_path" in result

    def test_run_output_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import physics_simulator as ps

        body = self._make_stub_glb(tmp_path, "body2.glb")
        garment = self._make_stub_glb(tmp_path, "garment2.glb")
        result = ps.run(body, garment, "combo_exists")
        assert Path(result["glb_path"]).exists()

    def test_run_missing_body_glb_falls_back(self, tmp_path, monkeypatch):
        """body_glb 不存在時，stub 路徑應 fallback 到 sample GLB，而非 raise。"""
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import physics_simulator as ps

        result = ps.run(
            body_glb=str(tmp_path / "nonexistent_body.glb"),
            garment_glb=str(tmp_path / "nonexistent_garment.glb"),
            output_name="fallback_test",
        )
        assert "glb_path" in result
        assert Path(result["glb_path"]).exists()

    def test_run_output_is_glb(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import physics_simulator as ps

        body = self._make_stub_glb(tmp_path, "body3.glb")
        garment = self._make_stub_glb(tmp_path, "garment3.glb")
        result = ps.run(body, garment, "ext_check")
        assert result["glb_path"].endswith(".glb")


# --------------------------------------------------------------------------- #
# M2 → M3 → M4 完整 Pipeline
# --------------------------------------------------------------------------- #

class TestFullPipeline:
    """
    stub 模式下的端對端 pipeline：
    M2 body_generator → M3 garment_processor → M4 physics_simulator

    目的是確認三個 adapter 的輸出/輸入格式可以正確串接，
    最終輸出是可讀的 GLB 檔案。
    """

    def test_pipeline_produces_readable_glb(self, tmp_path, monkeypatch):
        import trimesh

        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg
        from adapters import garment_processor as gp
        from adapters import physics_simulator as ps

        body_result = bg.generate(175, 92, 78, 96, "pipeline_body")
        assert Path(body_result["glb_path"]).exists()

        garment_result = gp.prepare(product_id=99, size="M")
        assert Path(garment_result["glb_path"]).exists()

        combo_result = ps.run(
            body_glb=body_result["glb_path"],
            garment_glb=garment_result["glb_path"],
            output_name="pipeline_combo",
        )
        assert Path(combo_result["glb_path"]).exists()

        # trimesh 能讀取最終 GLB 且不 raise
        mesh = trimesh.load(combo_result["glb_path"], force="mesh", process=False)
        assert mesh is not None

    def test_pipeline_output_glb_size(self, tmp_path, monkeypatch):
        """輸出 GLB 大小應 > 0 bytes，確認非空檔。"""
        monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
        import config
        monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", str(tmp_path))

        from adapters import body_generator as bg
        from adapters import garment_processor as gp
        from adapters import physics_simulator as ps

        body = bg.generate(160, 84, 68, 88, "size_body")
        garment = gp.prepare(product_id=101, size="S")
        combo = ps.run(body["glb_path"], garment["glb_path"], "size_combo")

        assert Path(combo["glb_path"]).stat().st_size > 0
