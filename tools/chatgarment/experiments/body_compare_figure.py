# -*- coding: utf-8 -*-
"""
SMPL 女性模板（舊尺寸檔）和 SMPL-X 中性（mean_all 尺寸檔）的比較圖，兩張：
  女性與中性_身體.png：兩個身體並排（正面、側面），同一個比例尺
  女性與中性_穿同一件.png：同一份 AI 結果（g03 藍帽 T、g05 米色長袖）分別在兩個身體上生成、新縫合

用簡單的平面著色畫三角形（畫家演算法：離相機遠的先畫），不需要 3D 繪圖套件。

跑法（在 newfashion_VT 根目錄，先跑過 sleeve_tpose_test.py、sleeve_smplx_test.py）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\body_compare_figure.py
輸出：output/chatgarment/ 底下兩張圖
"""
from pathlib import Path

import igl
import matplotlib
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BODIES = ROOT / "output" / "physics_export" / "_bodies"
OUT = ROOT / "output" / "chatgarment"
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False
FEMALE, NEUTRAL, SKIN = "#e07a72", "#5b8fd1", "#c8c8c8"
LIGHT = np.array([0.3, 0.5, 1.0]) / np.linalg.norm([0.3, 0.5, 1.0])
GARMENTS = [("g03_blue_hoodie", "g03 藍帽 T"), ("g05_beige_longsleeve_tee", "g05 米色長袖")]


def load(path, scale):
    v, f = igl.read_triangle_mesh(str(path))
    return v * scale, f


def view(v, side):
    """回傳 (畫面 x, 畫面 y, 深度)：正面看 +z 朝相機；側面看 +x 朝相機（人面向右）。"""
    return np.stack([v[:, 2], v[:, 1], v[:, 0]] if side else [v[:, 0], v[:, 1], v[:, 2]], 1)


def shade(ax, meshes, side=False):
    """meshes：[(頂點, 面, 顏色)]，全部一起依深度排序後畫，衣服和身體才會互相遮擋。
    顏色是一個顏色，或每個面一個 RGB（形狀 (面數, 3)）。"""
    polys, colors, depth = [], [], []
    for v, f, color in meshes:
        p = view(v, side)
        n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        lam = 0.35 + 0.65 * np.abs(n @ LIGHT)  # 衣服是單層面，正反面都打光
        rgb = np.asarray(color, float) if np.ndim(color) == 2 else np.array(to_rgb(color))
        polys.append(p[f][:, :, :2])
        colors.append(np.clip(lam[:, None] * rgb, 0, 1))
        depth.append(p[f][:, :, 2].mean(1))
    polys, colors, depth = np.concatenate(polys), np.concatenate(colors), np.concatenate(depth)
    order = np.argsort(depth)
    ax.add_collection(PolyCollection(polys[order], facecolors=colors[order], edgecolors="none"))
    ax.set_aspect("equal")
    ax.autoscale_view()


def main():
    female, ff = load(BODIES / "tpose_smpl_female.obj", 100)
    neutral, nf = load(BODIES / "smplx_neutral_theta0_feet0.obj", 100)
    hf, hn = female[:, 1].max(), neutral[:, 1].max()

    # 圖一：身體並排（正面左右錯開，側面前後錯開）
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [2.2, 1]})
    w = 195
    shade(axes[0], [(female - [w / 2, 0, 0], ff, FEMALE), (neutral + [w / 2, 0, 0], nf, NEUTRAL)])
    axes[0].set_title("正面（左：SMPL 女性模板　右：SMPL-X 中性）", fontsize=14)
    axes[0].text(-w / 2, hf + 3, f"女性 {hf:.0f} cm", ha="center", color=FEMALE, fontsize=12)
    axes[0].text(w / 2, hn + 3, f"中性 {hn:.0f} cm", ha="center", color=NEUTRAL, fontsize=12)
    shade(axes[1], [(female - [0, 0, 22], ff, FEMALE), (neutral + [0, 0, 22], nf, NEUTRAL)], side=True)
    axes[1].set_title("側面（左：女性　右：中性，面向右）", fontsize=14)
    for ax in axes:
        for h, c in ((hf, FEMALE), (hn, NEUTRAL)):
            ax.axhline(h, color=c, lw=1, ls=":")
        ax.set_ylim(-5, 185); ax.grid(alpha=0.25)
    fig.suptitle("交付身體：SMPL 女性模板 vs SMPL-X 中性", fontsize=17)
    plt.tight_layout()
    plt.savefig(OUT / "女性與中性_身體.png", dpi=75)
    plt.close(fig)

    # 圖二：穿同一件
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for row, (g, label) in enumerate(GARMENTS):
        for col, (path, body, bf, color, title) in enumerate((
                (ROOT / "output" / "sleeve_tpose" / f"new_{g}_sewn.obj", female, ff, FEMALE,
                 "女性模板＋舊尺寸檔（arm_length 80）"),
                (ROOT / "output" / "sleeve_smplx" / f"nocuff_{g}_sewn.obj", neutral, nf, NEUTRAL,
                 "SMPL-X 中性＋mean_all（arm_length 54）"))):
            ax = axes[row, col]
            gv, gf = load(path, 1)
            shade(ax, [(body, bf, SKIN), (gv, gf, color)])
            ax.set_xlim(-95, 95); ax.set_ylim(75, 180); ax.grid(alpha=0.25)
            ax.set_title(f"{label}　{title}", fontsize=13)
    fig.suptitle("同一份 AI 結果、同一種縫法（新縫合），只換身體與尺寸檔", fontsize=17)
    plt.tight_layout()
    plt.savefig(OUT / "女性與中性_穿同一件.png", dpi=75)
    plt.close(fig)

    # 圖三：女性身體也換新格式尺寸檔（sleeve_female_new_yaml.py 的結果）
    fy = ROOT / "output" / "sleeve_female_yaml"
    if not fy.exists():
        print("ok（沒有 sleeve_female_yaml 的結果，略過圖三）")
        return
    cols = [(ROOT / "output" / "sleeve_tpose" / "new_{g}_sewn.obj", female, ff, FEMALE, "女性＋舊 f_smpl_template（80）"),
            (fy / "mean_female_{g}_sewn.obj", female, ff, FEMALE, "女性＋新 mean_female（51.6）"),
            (fy / "mean_all_{g}_sewn.obj", female, ff, FEMALE, "女性＋新 mean_all（54）"),
            (ROOT / "output" / "sleeve_smplx" / "nocuff_{g}_sewn.obj", neutral, nf, NEUTRAL, "中性＋新 mean_all（54）")]
    fig, axes = plt.subplots(2, 4, figsize=(26, 10))
    for row, (g, label) in enumerate(GARMENTS):
        for col, (path, body, bf, color, title) in enumerate(cols):
            ax = axes[row, col]
            gv, gf = load(str(path).format(g=g), 1)
            shade(ax, [(body, bf, SKIN), (gv, gf, color)])
            ax.set_xlim(-92, 92); ax.set_ylim(75, 180); ax.grid(alpha=0.25)
            ax.set_title(f"{label}\n{title}", fontsize=13)
    fig.suptitle("尺寸檔（括號＝手臂長）決定袖長：女性身體換成新格式後，袖子和中性一樣", fontsize=17)
    plt.tight_layout()
    plt.savefig(OUT / "女性與中性_尺寸檔對比.png", dpi=70)
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
