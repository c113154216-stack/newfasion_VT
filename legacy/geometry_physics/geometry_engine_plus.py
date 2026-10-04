# -*- coding: utf-8 -*-
"""
geometry_engine.py 的延伸模組 —— 不修改 geometry_engine.py 本體（那是紀泓宇的模組），
這裡只是另外新增獨立的檔案，放這次 UV 貼圖驗證用到的計算邏輯。

跟 geometry_engine.py 的關係：只「使用」對方已經存在的函式（例如 load_obj_vf()、
_build_torso() 這些），不會反過來被 geometry_engine.py import，也不會動到它任何一行。

目前只有一個函式：compute_uv()，邏輯照搬 _uv_prototype_test.py 裡已經驗證過的版本，
沒有改動內容——已用 tshirt_base.obj + 棋盤格測試圖跑過端到端驗證，詳見
手冊/UV貼圖最小原型_操作指南.md 跟 手冊/UV貼圖_這兩天工作記錄.md。
"""
import numpy as np


def compute_uv(vertices: np.ndarray) -> np.ndarray:
    """
    三關卡 UV 邏輯，對服裝模型的頂點逐一分類、算座標。

    vertices: (N, 3) ndarray
    回傳: (N, 2) ndarray，範圍皆為 [0, 1]
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    is_front = z >= 0.0  # 關卡一：純幾何判斷，不碰照片

    uv = np.zeros((len(vertices), 2), dtype=np.float64)

    # 關卡二：正面的點做正交投影 —— 假裝從正前方拍照，只用 (x, y)，z（深度）在這步用不到
    front_x, front_y = x[is_front], y[is_front]
    x_min, x_max = front_x.min(), front_x.max()
    y_min, y_max = front_y.min(), front_y.max()
    uv[is_front, 0] = (x[is_front] - x_min) / (x_max - x_min)
    # 圖片座標系 V 通常是「上小下大」或相反，這裡讓 y 大（模型的上方）對應 V 大（貼圖的上方）
    uv[is_front, 1] = (y[is_front] - y_min) / (y_max - y_min)

    # 關卡三：非正面的點固定指向照片正中央（衣服本身通常置中滿版，抓到的是衣服真實顏色，不是背景）
    uv[~is_front] = [0.5, 0.5]

    return uv
