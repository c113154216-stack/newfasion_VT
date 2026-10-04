# -*- coding: utf-8 -*-
"""
UV 貼圖最小原型驗證（在這台裝置上重現另一台裝置「這兩天工作記錄」的結果）。

前綴底線 = 暫存測試腳本，完全不動任何生產程式碼
（geometry_engine.py / mesh_glb.py / garment_processor.py 一行都沒改）。

三關卡 UV 邏輯：
    關卡一：正不正面 —— 只看模型自己的 Z 座標（z >= 0 視為正面），跟照片無關
    關卡二：正面的點 —— 正交投影，直接拿 (x, y) 線性縮放到 0~1
    關卡三：非正面的點 —— 固定指向照片「正中央」（不是背景角落，避免抓到白底）

已知會踩的三個坑，這裡直接套用記錄裡驗證過的修法，不重新踩一次：
    1. trimesh 預設材質 metalness=1（全金屬）幾乎看不出貼圖顏色 -> 用 PBRMaterial 明講 metallicFactor=0
    2. 拿真實商品照片測試容易受圖片本身乾不乾淨干擾 -> 先用業界標準棋盤格測試圖排除變因
    3. 面朝向 (winding) 反了，外面看到的其實是被 culling 的內壁 -> mesh.fix_normals()
       （這招在純數學生成、頭尾開放的軀幹管狀網格上對記錄裡的情況沒用；
       但 tshirt_base.obj 是封閉網格，記錄裡驗證過在這個模型上有效，這裡直接用真正商品模型）
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

GEOMETRY_DIR = Path(__file__).resolve().parent / "geometry_physics"
sys.path.insert(0, str(GEOMETRY_DIR))
import geometry_engine as ge  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "uv_learning"
OUT_DIR.mkdir(exist_ok=True)
GLB_OUT = OUT_DIR / "step2_tshirt_uv_checker.glb"
TEXTURE_OUT = OUT_DIR / "step2_checkerboard.png"


def make_checkerboard(size=1024, cells=8):
    """業界標準棋盤格測試圖：高對比、每格顏色差異巨大，UV 有拉伸/扭曲一眼就看得出來。"""
    colors = [
        (255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 200, 0),
        (255, 0, 255), (0, 220, 220),
    ]
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cell = size // cells
    for i in range(cells):
        for j in range(cells):
            if (i + j) % 2 == 0:
                color = colors[(i * cells + j) % len(colors)]
                draw.rectangle(
                    [j * cell, i * cell, (j + 1) * cell - 1, (i + 1) * cell - 1],
                    fill=color,
                )
    # 外框 + 十字線 + 方向標記，方便判斷有沒有翻轉/旋轉
    draw.rectangle([0, 0, size - 1, size - 1], outline=(255, 255, 255), width=8)
    draw.line([(size // 2, 0), (size // 2, size)], fill=(255, 255, 255), width=6)
    draw.line([(0, size // 2), (size, size // 2)], fill=(255, 255, 255), width=6)
    draw.rectangle([0, 0, cell // 2, cell // 2], fill=(255, 255, 255))  # 左上角白方塊 = 方向標記
    img.save(TEXTURE_OUT)
    return img


def compute_uv(vertices):
    """
    三關卡 UV 邏輯，對 tshirt_base.obj 的頂點逐一分類、算座標。

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


def main():
    print("=== Step 2: UV 貼圖最小原型（真實商品模型 tshirt_base.obj）===")

    checker = make_checkerboard()
    print(f"棋盤格測試圖存到: {TEXTURE_OUT}")

    obj_path = Path(__file__).resolve().parent / "geometry_physics" / "assets" / "garments" / "tshirt_base.obj"
    vertices, faces = ge.load_obj_vf(str(obj_path))
    print(f"模型: {obj_path.name}，{len(vertices)} 頂點、{len(faces)} 面")

    uv = compute_uv(vertices)
    n_front = int((vertices[:, 2] >= 0.0).sum())
    print(f"正面頂點: {n_front} / {len(vertices)}（{n_front / len(vertices) * 100:.1f}%）")
    print(f"UV 範圍: U[{uv[:,0].min():.3f}, {uv[:,0].max():.3f}] "
          f"V[{uv[:,1].min():.3f}, {uv[:,1].max():.3f}]")

    # 修法 1：明講 PBR 材質參數，metallicFactor=0 避免全金屬看不出貼圖顏色
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=checker,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

    # 修法 3：面朝向反了會導致外壁被 culling、看起來像整片單一顏色。
    # tshirt_base.obj 是封閉網格，這招在這個模型上記錄裡驗證過有效。
    mesh.fix_normals()

    mesh.export(str(GLB_OUT))
    print(f"GLB 匯出完成: {GLB_OUT}")
    print("=== 完成，下一步：丟進 Three.js 檢視器確認棋盤格真的貼在外壁上 ===")


if __name__ == "__main__":
    main()
