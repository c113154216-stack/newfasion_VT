# -*- coding: utf-8 -*-
"""
SMPL-X 身體（θ=0、手臂水平），和紀泓宇的物理引擎用同一個身體、同一個座標系。

- 用 SMPL-X 權重檔直接算：頂點 ＝ v_template ＋ shapedirs · betas（θ=0 所以沒有姿勢變形），
  只要 numpy，GarmentCode 的 .venv 就能跑。
- 座標：SMPL-X 原始座標（公尺，+y 上、+z 前，骨盆在原點附近，腳底不在 0），和紀泓宇一樣。
- 已核對：betas 全 0 時和紀泓宇給的 smplx_female_theta0.obj 每個頂點差 < 1e-6 m。

SMPL-X 權重的授權禁止散布，生成的身體網格也只放在 output/（不上傳）。
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SMPLX_DIR = REPO / "legacy" / "geometry_physics" / "assets" / "smplx"
NECK, L_COLLAR, R_COLLAR = 12, 13, 14  # SMPL-X 關節編號（和 SMPL 相同）


def load_model(gender="female"):
    return np.load(SMPLX_DIR / f"SMPLX_{gender.upper()}.npz", allow_pickle=True)


def make_body(gender="female", betas=None):
    """回傳 (頂點 (10475,3) 公尺、面、蒙皮權重 (10475,55))，SMPL-X 原始座標。"""
    model = load_model(gender)
    v = model["v_template"].astype(np.float64)
    if betas is not None and len(betas):
        betas = np.asarray(betas, dtype=np.float64)
        v = v + model["shapedirs"][:, :, : len(betas)] @ betas
    return v, model["f"].astype(np.int64), model["weights"]


def from_measurement_json(path):
    """紀泓宇的量測 JSON（gender_model、betas）→ 身體。"""
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    return make_body(info["gender_model"], info["betas"])
