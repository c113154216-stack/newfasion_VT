# -*- coding: utf-8 -*-
r"""把 spec_sizes.py 做好的各尺碼「攤平」成 2D 並排／疊在一起比較，像商品平拍照那樣。

用 2D 版片直接排，不是從 3D 壓扁（壓扁要再模擬一次）：
  左：平放輪廓 —— 前片左右兩半拼成衣身，寬度縮放成「平放胸寬」（真的衣服平放時，摺在側邊的是前後片的平均寬度），
      袖片接在肩點上，往下斜 35°（只是為了像平拍照，角度不影響尺寸）。
  右：版片本身 —— 後片右半、袖片，三個尺碼疊在一起（GarmentCode 原始座標，下襬中心對齊）。

跑法（在 newfashion_VT 根目錄，先跑過 spec_sizes.py）：
    .venv\Scripts\python flows\size_chart\spec_flatlay.py [網路衣服資料夾] [尺碼，預設 S,M,L]
輸出：output/spec_to_pattern/<資料夾>/平放_<S-M-L>.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
BODY_FIT = HERE.parent / "body_fit"  # 共用的打版、縫合、匯出程式（照身體生成那條流程）
sys.path.insert(0, str(BODY_FIT))
sys.path.insert(0, str(BODY_FIT / "experiments"))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402

ROOT = sp.ROOT
COLORS = {"XXS": "#cfe0f2", "XS": "#b5d0ec", "S": "#e39b3b", "M": "#2f7fc1", "L": "#3c8d4f", "XL": "#7a4fa0"}
SLEEVE_DROP_DEG = 35


def outline(edges):
    """版片各邊取樣點接成一圈。"""
    return np.vstack(edges)


def rot(p, deg):
    a = np.deg2rad(deg)
    return p @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def flatlay(pattern, chest):
    """回傳 [(多邊形, 種類)]，座標：肩頸點（衣服最高點）在 y=0、中心在 x=0，往下是負。"""
    P = pattern["panels"]
    _, fe = sp.sample_panel(P["right_ftorso"])
    front = outline(fe)
    # 前片右半在 x ≤ 0；腋下高度的寬度縮放成平放胸寬的一半
    y0 = 0.55 * np.ptp(front[:, 1]) + front[:, 1].min()
    k = (chest / 2) / sp.width_at(fe, y0)
    front = front * [k, 1]
    top = front[:, 1].max()
    bv = np.asarray(P["right_ftorso"]["vertices"], float) * [k, 1]
    tip_y_cands = bv[bv[:, 1] > bv[:, 1].max() - 8]
    tip = tip_y_cands[np.argmin(tip_y_cands[:, 0])]  # 肩點：肩線上最外側（x 最小）
    # 袖片：上沿 (0,0)→(L,0)，袖山頂在 x=L、袖口在 x=0；翻成往左延伸、袖山頂接在肩點
    _, se = sp.sample_panel(P["right_sleeve_f"])
    sl = outline(se)
    L = np.asarray(P["right_sleeve_f"]["vertices"], float)[:, 0].max()
    sl = np.c_[sl[:, 0] - L, sl[:, 1]]  # 袖山頂移到原點，袖子往 -x
    sl = rot(sl, SLEEVE_DROP_DEG) + tip  # 往下斜（-x 方向往下轉）
    polys = []
    for s in (1, -1):  # 右半、左半（鏡像）
        polys.append((np.c_[front[:, 0] * s, front[:, 1] - top], "torso"))
        polys.append((np.c_[sl[:, 0] * s, sl[:, 1] - top], "sleeve"))
    return polys


def main(folder=None, sizes="S,M,L"):
    folder = Path(folder) if folder else HERE / "網路衣服_尺寸資料" / "z06_長袖T"
    if not folder.is_absolute():
        folder = ROOT / folder
    sizes = sizes.split(",")
    out = ROOT / "output" / "spec_to_pattern" / folder.name
    sp.dm.ensure_system_json()
    _, body_yaml, _ = prepare_smplx_body("neutral")
    sp.BASE_BODY = BodyParameters(str(body_yaml))

    items = []
    for s in sizes:
        d = out / "尺碼" / s
        design = yaml.safe_load((d / "design.yaml").read_text(encoding="utf-8"))["design"]
        meas = json.loads((d / "量測.json").read_text(encoding="utf-8"))
        items.append((s, sp.build(design, {}), meas["版片量到"], meas["尺寸表"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(19, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.1, 1.25])
    ax0, ax1, ax2 = (fig.add_subplot(g) for g in gs)
    img = folder / f"{folder.name}_front.jpg"
    if img.exists():
        ax0.imshow(mpimg.imread(str(img)))
    ax0.set_title("商品照（平拍）", fontsize=13)
    ax0.axis("off")

    # 中：平放輪廓，大的在下面
    order = sorted(items, key=lambda it: -it[2]["length"])
    for s, pat, m, t in order:
        for poly, kind in flatlay(pat, m["chest"]):
            ax1.add_patch(Polygon(poly, closed=True, facecolor=COLORS.get(s, "#888"), edgecolor="white", lw=0.6, alpha=0.95))
    for i, (s, pat, m, t) in enumerate(items):
        ax1.text(0.01, 0.96 - 0.065 * i, f"{s}：胸寬 {m['chest']:.0f}　衣長 {m['length']:.0f}　袖長 {m['sleeve']:.0f}",
                 transform=ax1.transAxes, fontsize=12, va="top", color=COLORS.get(s, "#888"), fontweight="bold")
    ax1.set_aspect("equal")
    ax1.autoscale_view()
    ax1.set_title("攤平疊在一起（由版片排出，袖子斜 35° 只是為了像平拍照）", fontsize=13)
    ax1.set_xlabel("cm")
    ax1.grid(alpha=0.25)

    # 右：版片本身
    for s, pat, m, t in order:
        for name, dx in [("right_btorso", 0), ("right_sleeve_f", 6)]:
            _, es = sp.sample_panel(pat["panels"][name])
            poly = outline(es)
            if "sleeve" in name:
                poly = np.c_[poly[:, 0] + dx, poly[:, 1] + 80]
            ax2.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=COLORS.get(s, "#888"), lw=1.8))
    ax2.set_aspect("equal")
    ax2.autoscale_view()
    ax2.set_title("版片疊在一起（後片右半、袖片）", fontsize=13)
    ax2.set_xlabel("cm")
    ax2.grid(alpha=0.25)
    ax2.legend(handles=[Patch(color=COLORS.get(s, "#888"), label=s) for s, *_ in items], loc="lower right", fontsize=11)

    fig.suptitle(f"{folder.name}：{'、'.join(sizes)} 攤平比較（胸寬、衣長、袖長照尺寸表，單位 cm）", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = out / f"平放_{'-'.join(sizes)}.png"
    fig.savefig(png, dpi=110)
    print("輸出：", png)


if __name__ == "__main__":
    main(*sys.argv[1:])
