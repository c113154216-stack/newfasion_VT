# -*- coding: utf-8 -*-
"""
UV 貼圖學習 - Step 1：幫 _build_torso() 算 UV 座標，並「畫出來」驗證

目的：在碰真正的貼圖之前，先確認「theta/2π 當 U、t 當 V」這個想法是對的。
不修改 geometry_engine.py 本體（那是 Chi 的模組），這裡只是呼叫既有函式做實驗。

跑法：
    python uv_learning/step1_compute_uv.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# geometry_physics 底下的 geometry_engine.py 用的是相對 import 風格，
# 要先把資料夾加進 sys.path 才能 import 到（跟 garment_processor.py 的做法一致）
GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "geometry_physics"
sys.path.insert(0, str(GEOMETRY_DIR))

import geometry_engine as ge  # noqa: E402


def compute_torso_uv():
    """
    重新算一次跟 _build_torso() 內部一模一樣的 T, THETA 網格，
    藉此推出跟頂點一一對應的 UV 座標。

    為什麼要「重算」而不是直接改 _build_torso() 回傳 UV？
    因為現在只是驗證想法對不對，還不確定要不要真的動 Chi 的模組，
    所以先在外部用同樣的公式重建一份，兩邊互不影響。
    """
    hs = ge.TORSO_HEIGHT_SEGMENTS   # 48，高度方向切幾段
    rs = ge.TORSO_RADIAL_SEGMENTS   # 48，繞一圈切幾段

    t = np.arange(hs + 1) / hs                      # 0(下擺) ~ 1(肩線)
    theta = (np.arange(rs + 1) / rs) * 2.0 * np.pi   # 0 ~ 2π，繞身體一圈的角度

    # 這裡的 indexing='ij' 要跟 _build_torso() 內部完全一致，
    # 不然算出來的 UV 會跟頂點順序對不上，貼出來的圖會亂掉
    T, THETA = np.meshgrid(t, theta, indexing='ij')

    u = THETA.ravel() / (2.0 * np.pi)   # 角度 0~2π 直接線性映射到 U: 0~1
    v = T.ravel()                        # 高度本來就是 0~1，直接當 V

    uv = np.stack([u, v], axis=1)
    return uv


def main():
    vertices, weights, faces = ge._build_torso()
    uv = compute_torso_uv()

    # 第一個要驗證的事：UV 的數量必須跟頂點數量完全一樣，
    # 一個頂點對應一組 (u, v)，缺一不可
    print(f"頂點數: {len(vertices)}")
    print(f"UV 數量: {len(uv)}")
    assert len(vertices) == len(uv), "UV 數量跟頂點數量對不上，公式一定哪裡錯了"
    print("OK: UV 數量跟頂點數量一致")
    print(f"U 範圍: [{uv[:, 0].min():.3f}, {uv[:, 0].max():.3f}]")
    print(f"V 範圍: [{uv[:, 1].min():.3f}, {uv[:, 1].max():.3f}]")

    fig = plt.figure(figsize=(13, 6))

    # 左圖：3D 網格，用 U 值上色 —— 讓你「看見」UV 是怎麼繞著身體纏一圈的
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc1 = ax1.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1],
                       c=uv[:, 0], cmap='hsv', s=3)
    ax1.set_title("3D 軀幹網格（顏色 = U 值，繞一圈剛好經過整個色環）")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y（高度）")
    fig.colorbar(sc1, ax=ax1, shrink=0.6, label="U")

    # 右圖：把同一批頂點畫在 (U, V) 平面上 —— 這就是理論文件說的「UV Layout」，
    # 一個攤平的正方形參照空間，此刻還沒貼圖，只是輪廓/分布
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(uv[:, 0], uv[:, 1], c=uv[:, 0], cmap='hsv', s=3)
    ax2.set_title("UV Layout（攤平後的樣子）")
    ax2.set_xlabel("U")
    ax2.set_ylabel("V")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_aspect('equal')
    ax2.axvline(0, color='red', linestyle='--', linewidth=0.8, label='seam (U=0)')
    ax2.axvline(1, color='red', linestyle='--', linewidth=0.8, label='seam (U=1)')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = Path(__file__).parent / "step1_uv_layout.png"
    plt.savefig(out_path, dpi=130)
    print(f"圖存到: {out_path}")


if __name__ == "__main__":
    main()
