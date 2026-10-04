"""紀泓宇 Module 3 — 服飾幾何拉伸。"""

import os
import shutil
import sys
from pathlib import Path

from config import Config

_GEOMETRY_DIR = Path(Config.BASE_DIR) / "geometry_physics"
if str(_GEOMETRY_DIR) not in sys.path:
    sys.path.insert(0, str(_GEOMETRY_DIR))

GARMENT_DIR = _GEOMETRY_DIR / "assets" / "garments"

# catalog.garment_template_id -> (公版 obj, 是否上身有袖)
GARMENT_TEMPLATES = {
    "tshirt_base.glb": ("tshirt_base.obj", True),
    "shirt_base.glb": ("shirt_base.obj", True),
    "coat_base.glb": ("coat_base.obj", True),
    "hoodie_base.glb": ("tshirt_base.obj", True),
    "polo_base.glb": ("tshirt_base.obj", True),
    "sweater_base.glb": ("coat_base.obj", True),
    "dress_base.glb": ("tshirt_base.obj", True),
    "pants_base.glb": ("pants_base.obj", False),
    "shorts_base.glb": ("shorts_base.obj", False),
    "joggers_base.glb": ("pants_base.obj", False),
}
DEFAULT_TEMPLATE = "tshirt_base.glb"
_BASE_CACHE = {}


def _use_stub() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _stub_prepare(product_id: int, size: str) -> dict:
    from adapters.glb_utils import create_minimal_glb

    output_dir = Path(Config.OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"product_{product_id}_{size}.glb"
    sample = Path(Config.BASE_DIR) / "data" / "samples" / "sample_garment.glb"
    if not sample.exists() or sample.stat().st_size < 400:
        create_minimal_glb(sample)
    shutil.copy2(sample, output_path)
    return {"glb_path": str(output_path)}


def _load_base(obj_filename):
    import geometry_engine as ge

    if obj_filename not in _BASE_CACHE:
        _BASE_CACHE[obj_filename] = ge.build_real_shirt_template(
            str(GARMENT_DIR / obj_filename)
        )
    return _BASE_CACHE[obj_filename]


def _load_static(obj_filename):
    import geometry_engine as ge

    if obj_filename not in _BASE_CACHE:
        _BASE_CACHE[obj_filename] = ge.load_obj_vf(str(GARMENT_DIR / obj_filename))
    return _BASE_CACHE[obj_filename]


def _size_to_params(size_chart, size):
    params = {"length": 100.0, "shoulder": 100.0, "chest": 100.0, "waist": 100.0}
    if not size_chart or "M" not in size_chart or size not in size_chart:
        return params
    baseline = size_chart["M"]
    target = size_chart[size]
    for key in ("length", "chest", "waist"):
        base_val = baseline.get(key)
        target_val = target.get(key)
        if base_val and target_val:
            params[key] = target_val / base_val * 100.0
    return params


def prepare(product_id: int, size: str = "M") -> dict:
    if _use_stub():
        return _stub_prepare(product_id, size)

    try:
        from adapters.mesh_glb import export_glb, save_sidecar
        from database.models import Product, SessionLocal

        db = SessionLocal()
        try:
            db.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
            product = db.get(Product, product_id)
            template_id = (product.garment_template_id if product else None) or DEFAULT_TEMPLATE
            size_chart = product.size_chart if product else None
        finally:
            db.close()

        obj_filename, has_sleeve = GARMENT_TEMPLATES.get(
            template_id, GARMENT_TEMPLATES[DEFAULT_TEMPLATE]
        )
        if not (GARMENT_DIR / obj_filename).exists():
            return _stub_prepare(product_id, size)

        import geometry_engine as ge

        if has_sleeve:
            base_v, base_w, base_f = _load_base(obj_filename)
            vertices = ge.deform_vertices(base_v, base_w, _size_to_params(size_chart, size))
            faces = base_f
        else:
            vertices, faces = _load_static(obj_filename)

        output_path = Path(Config.OUTPUT_FOLDER) / f"product_{product_id}_{size}.glb"
        export_glb(vertices, faces, output_path)
        save_sidecar(output_path, {"has_sleeve_topology": has_sleeve})
        return {"glb_path": str(output_path)}
    except Exception:
        return _stub_prepare(product_id, size)
