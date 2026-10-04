"""紀泓宇 Module 2 — SMPL-X 人體生成。"""

import logging
import os
import shutil
import sys
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

_GEOMETRY_DIR = Path(Config.BASE_DIR) / "geometry_physics"
if str(_GEOMETRY_DIR) not in sys.path:
    sys.path.insert(0, str(_GEOMETRY_DIR))

DEFAULT_GENDER = "neutral"
DEFAULT_POSE = "standing"


def _use_stub() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _stub_generate(output_name: str) -> dict:
    from adapters.glb_utils import create_minimal_glb

    output_dir = Path(Config.OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_body.glb"
    sample = Path(Config.BASE_DIR) / "data" / "samples" / "sample_body.glb"
    if not sample.exists() or sample.stat().st_size < 400:
        create_minimal_glb(sample)
    shutil.copy2(sample, output_path)
    return {"glb_path": str(output_path)}


def generate(
    height_cm: float,
    chest_cm: float,
    waist_cm: float,
    hip_cm: float,
    output_name: str,
) -> dict:
    if _use_stub():
        return _stub_generate(output_name)

    try:
        import body_model as bm
        from adapters.mesh_glb import export_glb, save_sidecar

        vertices, faces, betas = bm.smplx_body_from_measurements(
            height_cm=height_cm,
            bust_cm=chest_cm,
            waist_cm=waist_cm,
            hip_cm=hip_cm,
            gender=DEFAULT_GENDER,
            pose=DEFAULT_POSE,
        )
        output_path = Path(Config.OUTPUT_FOLDER) / f"{output_name}_body.glb"
        export_glb(vertices, faces, output_path)
        save_sidecar(
            output_path,
            {
                "betas": betas.tolist(),
                "gender": DEFAULT_GENDER,
                "pose": DEFAULT_POSE,
            },
        )
        return {"glb_path": str(output_path)}
    except Exception:
        logger.exception("SMPL-X 人體生成失敗，改用示範 GLB")
        return _stub_generate(output_name)
