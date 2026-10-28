# -*- coding: utf-8 -*-
"""
GarmentCode 領口形狀參數的示意圖：7 種形狀、fc_angle、bezier 控制點、船領。
直接呼叫 GarmentCode 的形狀函式（collars.py），畫出前片的領口線（半邊算出來後左右鏡射）。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\neck_shape_demo.py
輸出：output/chatgarment/領口參數示意.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "GarmentCodeRC"))
from assets.garment_programs import collars as C  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False
DEPTH, WIDTH = 8.0, 20.0  # 一般圓領 T：前領深 8 cm、領口寬約等於頸寬 20 cm


def half(shape, depth=DEPTH, width=WIDTH, **kw):
    """回傳半邊領口線的點：從頸側點 (0, 0) 到前中心 (width/2, -depth)。"""
    edges = getattr(C, shape)(depth, width, verbose=False, **kw)
    pts = []
    for e in edges:
        curve = e.as_curve()
        pts += [(curve.point(t).real, curve.point(t).imag) for t in np.linspace(0, 1, 40)]
    return np.array(pts)


def draw(ax, pts, width=WIDTH, title="", color="#1f4e9e", ctrl=None):
    """把半邊鏡射成整個領口，座標改成以前中心為 0。"""
    x, y = pts[:, 0] - width / 2, pts[:, 1]
    ax.plot(np.r_[x, -x[::-1]], np.r_[y, y[::-1]], color=color, lw=2.5)
    ax.axhline(0, color="#999", lw=1, ls="--")  # 肩線高度（頸側點）
    ax.plot([-width / 2, width / 2], [0, 0], "o", color="#d62728", ms=5)
    if ctrl is not None:
        ax.plot(*ctrl, "x", color="#ff7f0e", ms=10, mew=2.5)
    ax.set_xlim(-19, 19); ax.set_ylim(-18, 3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([-15, -10, -5, 0])
    ax.set_title(title, fontsize=12)


def bezier_ctrl(x, y, flip, depth=DEPTH, width=WIDTH):
    """CurveEdge 的控制點是相對座標：x 沿著頸側點→前中心那條直線，y 垂直它（乘以直線長度）。"""
    a, b = np.array([0.0, 0.0]), np.array([width / 2, -depth])
    d = b - a
    n = np.array([-d[1], d[0]])
    p = a + x * d + (1 if flip else -1) * y * n
    return p[0] - width / 2, p[1]


def main():
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))

    # 第 1 列：7 種形狀（寬 20、深 8）
    shapes = [("CircleNeckHalf", "圓"), ("CurvyNeckHalf", "彎曲圓弧"), ("CircleArcNeckHalf", "圓弧（angle 90）"),
              ("Bezier2NeckHalf", "貝茲（x 0.3、y 0.55）"), ("VNeckHalf", "V")]
    for ax, (s, name) in zip(axes[0], shapes):
        kw = {"x": 0.3, "y": 0.55} if s == "Bezier2NeckHalf" else {"angle": 90} if s == "CircleArcNeckHalf" else {}
        draw(ax, half(s, **kw), title=f"{name}\n{s}")
    # 第 2 列：方、梯形，以及 fc_angle 對梯形的影響
    draw(axes[1, 0], half("SquareNeckHalf"), title="方\nSquareNeckHalf")
    for ax, ang in zip(axes[1, 1:], (70, 90, 100, 110)):
        draw(ax, half("TrapezoidNeckHalf", angle=ang), title=f"梯形 fc_angle＝{ang}°", color="#2ca02c")

    # 第 3 列：fc_angle 對圓弧的影響＋貝茲控制點 x
    for ax, ang in zip(axes[2, :2], (70, 110)):
        draw(ax, half("CircleArcNeckHalf", angle=ang), title=f"圓弧 fc_angle＝{ang}°", color="#2ca02c")
    for ax, bx in zip(axes[2, 2:], (0.1, 0.5, 0.9)):
        draw(ax, half("Bezier2NeckHalf", x=bx, y=0.3), title=f"貝茲 x＝{bx}（y＝0.3）", color="#8a5fa8",
             ctrl=bezier_ctrl(bx, 0.3, False))

    # 第 4 列：貝茲控制點 y、flip，以及船領
    for ax, by in zip(axes[3, :2], (0.1, 0.6)):
        draw(ax, half("Bezier2NeckHalf", x=0.5, y=by), title=f"貝茲 y＝{by}（x＝0.5）", color="#8a5fa8",
             ctrl=bezier_ctrl(0.5, by, False))
    draw(axes[3, 2], half("Bezier2NeckHalf", x=0.5, y=0.3, flip=True), title="貝茲 flip＝True（曲線反過來）",
         color="#8a5fa8", ctrl=bezier_ctrl(0.5, 0.3, True))
    boat_w, boat_d = 32.0, 3.0
    draw(axes[3, 3], half("CircleNeckHalf", depth=DEPTH), title="一般圓領：寬 20、深 8", color="#1f4e9e")
    draw(axes[3, 4], half("Bezier2NeckHalf", depth=boat_d, width=boat_w, x=0.5, y=0.1), width=boat_w,
         title=f"船領：寬 {boat_w:.0f}、深 {boat_d:.0f}\n（collar.width 大、領深淺）", color="#d62728")

    fig.text(0.5, 0.005, "灰色虛線＝肩線高度　紅點＝頸側點（領口兩端）　橘色 ×＝貝茲控制點　單位 cm，前中心在 0",
             ha="center", fontsize=13)
    fig.suptitle("GarmentCode 前領口形狀與參數（從正面看前片）", fontsize=17)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    dst = ROOT / "output" / "chatgarment" / "領口參數示意.png"
    plt.savefig(dst, dpi=65)
    print(dst)


if __name__ == "__main__":
    main()
