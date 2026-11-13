# -*- coding: utf-8 -*-
"""
實驗五：整件往上滑，到底是哪裡撐不開、身體在哪裡往上推？（接實驗四 slide_fix_test.py）

用現在的交付縫法重縫一次（版片三角化每次產生都不同，不能沿用實驗四的 obj），算兩件事：
  1. 每條邊被拉長／壓短多少：碰到 ±2% 限制的邊（權重變 100）就是「撐不開」的地方
  2. 身體推衣服的力：縫好時每個頂點受力平衡，貼著身體的頂點，布料的合力（邊＋彎曲）就是被身體抵住的力，
     身體的推力 ＝ −布料合力。取 y 分量就是「往上推多少」，再依最近身體面的 SMPL-X 主導關節分區加總。

跑法（在 newfashion_VT 根目錄，先跑過 slide_fix_test.py）：
    .venv\\Scripts\\python flows\\body_fit\\experiments\\where_push.py [衣服]
輸出：output/chatgarment/上滑位置_<衣服>.png、output/slide_fix/where_<衣服>.json
"""
import json
import pickle
import sys
from pathlib import Path

import igl
import matplotlib
import numpy as np
import scipy.sparse as sp
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors as mcolors  # noqa: E402

import design_to_mesh as dm  # noqa: E402
import smplx_body  # noqa: E402
from body_compare_figure import SKIN, shade  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import RESOLUTION, SEW  # noqa: E402
from sew import mesh_edges, rest_lengths, sew  # noqa: E402
from slide_fix_test import OUT, SRC  # noqa: E402

DST = HERE.parents[2] / "output" / "chatgarment"
CONTACT_CM = 1.0  # 離身體 1 cm 內算「貼著身體」（縫合推出距離是 0.5 cm）
REGIONS = {  # SMPL-X 關節 → 區域
    "脖子": [12, 15], "鎖骨／肩頸": [13, 14], "肩膀": [16, 17], "上臂": [18, 19], "前臂／手": [20, 21],
    "胸口（spine3）": [9], "腹部（spine1-2）": [3, 6], "骨盆": [0, 1, 2],
}


def forces(x, E, rest, f, bend):
    """縫好狀態下布料（邊＋彎曲）對每個頂點的合力；權重照縫合時的規則（超過 ±2% 的邊權重 100）。"""
    n = len(x)
    d = x[E[:, 0]] - x[E[:, 1]]
    length = np.linalg.norm(d, axis=1)
    ratio = length / rest
    w = np.where((ratio > 1 + SEW["stretch_limit"]) | (ratio < 1 - SEW["compress_limit"]), SEW["stretch_weight"], 1.0)
    pull = (w * (length - rest) / length)[:, None] * d  # 邊太長 → 把兩端拉近
    F = np.zeros_like(x)
    np.add.at(F, E[:, 0], -pull)
    np.add.at(F, E[:, 1], pull)
    adj = sp.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    Lap = sp.identity(n) - sp.diags(1.0 / np.maximum(deg, 1)) @ adj
    F -= bend * (Lap.T @ (Lap @ x))
    return F, ratio, w


def face_colors(values, cmap, norm):
    return cmap(norm(values))[:, :3]


def main(g="g05_beige_longsleeve_tee"):
    dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body()
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv *= 100
    weights = smplx_body.make_body("neutral")[2]
    joint = np.asarray(weights).argmax(axis=1)

    design = yaml.safe_load((SRC / g / "design_used.yaml").read_text(encoding="utf-8"))["design"]
    name = f"slide_{g}"
    _, _, paths = dm.generate(design, body_yaml, name, resolution=RESOLUTION)
    folder = paths.g_box_mesh.parent
    v, f = igl.read_triangle_mesh(str(paths.g_box_mesh))
    E = mesh_edges(f)
    rest = rest_lengths(v, E, pickle.loads(paths.g_orig_edge_len.read_bytes()))
    seg = np.array((folder / f"{name}_sim_segmentation.txt").read_text().splitlines())
    # 重新產生版片時三角化不固定，不能沿用實驗四存的 obj，這裡用同一份版片重縫（現在的交付縫法）
    x = sew(v, E, rest, body=(bv, bf), **SEW)
    igl.write_triangle_mesh(str(OUT / f"where_{g}_sewn.obj"), x, f)

    F, ratio, w = forces(x, E, rest, f, SEW["bend"])
    d2, face, _ = igl.point_mesh_squared_distance(x, bv, bf)
    contact = np.sqrt(d2) < CONTACT_CM
    push = np.where(contact[:, None], -F, 0.0)  # 身體推衣服的力
    vj = joint[bf[face]]  # 最近身體面三個角的主導關節，取第一個角
    region_of = {j: r for r, js in REGIONS.items() for j in js}
    vregion = np.array([region_of.get(int(j), "其他") for j in vj[:, 0]])

    # ---- 數字 ----
    up_total = push[:, 1][push[:, 1] > 0].sum()
    by_region = {}
    for r in list(REGIONS) + ["其他"]:
        m = vregion == r
        if m.any():
            by_region[r] = {"貼身頂點": int((m & contact).sum()),
                            "往上推力": round(float(push[m, 1].clip(min=0).sum()), 1),
                            "往上推力佔%": round(100 * float(push[m, 1].clip(min=0).sum() / up_total), 1),
                            "往下推力": round(float(push[m, 1].clip(max=0).sum()), 1)}
    vpanel = np.array([s.split("_")[0] if s else "?" for s in seg])
    panels = {}
    for p in sorted(set(seg)):
        em = (seg[E[:, 0]] == p) & (seg[E[:, 1]] == p)
        if em.sum() < 20:
            continue
        panels[p] = {"邊數": int(em.sum()), "拉長超過2%的邊%": round(100 * float(np.mean(ratio[em] > 1.02)), 1),
                     "壓短超過2%的邊%": round(100 * float(np.mean(ratio[em] < 0.98)), 1),
                     "平均拉長%": round(100 * float(ratio[em].mean() - 1), 2)}
    seam = seg[E[:, 0]] != seg[E[:, 1]]
    panels["（縫線上的邊）"] = {"邊數": int(seam.sum()), "拉長超過2%的邊%": round(100 * float(np.mean(ratio[seam] > 1.02)), 1),
                          "壓短超過2%的邊%": round(100 * float(np.mean(ratio[seam] < 0.98)), 1),
                          "平均拉長%": round(100 * float(ratio[seam].mean() - 1), 2)}
    res = {"往上推力合計": round(float(up_total), 1), "往下推力合計": round(float(push[:, 1].clip(max=0).sum()), 1),
           "依身體區域": by_region, "依版片": panels}
    (OUT / f"where_{g}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 圖 ----
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fr = np.zeros(len(f))  # 每個面三條邊裡偏離最多的拉長量（%）
    key = {tuple(e): k for k, e in enumerate(E)}
    for k, (a, b, c) in enumerate(f):
        r = [ratio[key[tuple(sorted(p))]] - 1 for p in ((a, b), (b, c), (a, c))]
        fr[k] = 100 * r[int(np.argmax(np.abs(r)))]
    fp = push[f, 1].sum(axis=1)  # 每個面的往上推力
    cm1, n1 = matplotlib.colormaps.get_cmap("coolwarm"), mcolors.TwoSlopeNorm(0, -6, 6)
    lim = np.percentile(np.abs(fp[fp != 0]), 95)
    cm2, n2 = matplotlib.colormaps.get_cmap("PuOr_r"), mcolors.TwoSlopeNorm(0, -lim, lim)
    back = np.array([-1, 1, -1])
    views = [("正面", 1, False), ("背面", back, False), ("側面（人面向右）", 1, True)]
    fig, axes = plt.subplots(2, 3, figsize=(21, 15))
    for row, (vals, cmap, norm, title) in enumerate(
            [(fr, cm1, n1, "邊拉長 %（紅＝拉長、藍＝壓短；超過 ±2% 就變硬＝撐不開）"),
             (fp, cm2, n2, "身體推衣服的力（橘＝往上推、紫＝往下推）")]):
        for col, (vt, s, side) in enumerate(views):
            ax = axes[row, col]
            gcol = face_colors(vals, cmap, norm)
            shade(ax, [(bv * s, bf, SKIN), (x * s, f, gcol)], side=side)
            ax.set_ylim(78, 178)
            ax.set_xlim(-60, 60) if side else ax.set_xlim(-92, 92)
            ax.set_title(vt, fontsize=13)
            ax.grid(alpha=0.2)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes[row].tolist(), shrink=0.7, pad=0.01).ax.tick_params(labelsize=10)
        axes[row, 0].set_ylabel(title, fontsize=13)
    top = sorted(by_region.items(), key=lambda kv: -kv[1]["往上推力"])[:4]
    txt = "往上推力來源：" + "、".join(f"{r} {d['往上推力佔%']:.0f}%" for r, d in top)
    fig.suptitle(f"{g}　哪裡撐不開、身體在哪裡往上推（SMPL-X 中性 T-pose，現在的交付縫法）\n{txt}", fontsize=16)
    plt.savefig(DST / f"上滑位置_{g.split('_')[0]}.png", dpi=60, bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
