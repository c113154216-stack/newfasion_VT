# -*- coding: utf-8 -*-
r"""照尺寸表一次做好幾個尺碼（預設 S、M、L），縫到 SMPL-X 中性身體上並排比較。

肩寬不照尺寸表（肩線往外推會縫壞，見 spec_sew_compare.py），其餘胸寬、衣長、袖長照尺寸表：
  胸寬 → shirt.width 公式直接算；衣長、袖長 → 在 2D 版片上量、按比例修到差 0.2 cm 內。

跑法（在 newfashion_VT 根目錄）：
    .venv\Scripts\python tools\chatgarment\experiments\spec_sizes.py [網路衣服資料夾] [尺碼，逗號分隔，預設 S,M,L]
輸出：output/spec_to_pattern/<資料夾>/尺碼_<S-M-L>.png、尺碼/<尺碼>/（design.yaml、縫好的 .obj、量測.json）
"""
import json
import shutil
import sys
from pathlib import Path

import igl
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
from assets.bodies.body_params import BodyParameters  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import build  # noqa: E402
from body_compare_figure import SKIN, shade, plt  # noqa: E402

ROOT = sp.ROOT
COLORS = {"S": "#9cc3e6", "M": "#5b8fd1", "L": "#2f5f9e"}
NAMES = {"chest": "胸寬", "shoulder": "肩寬", "length": "衣長", "sleeve": "袖長"}


def main(folder=None, sizes="S,M,L"):
    folder = Path(folder) if folder else ROOT / "網路衣服_尺寸資料" / "z06_長袖T"
    if not folder.is_absolute():
        folder = ROOT / folder
    sizes = sizes.split(",")
    info = json.loads((folder / "尺寸.json").read_text(encoding="utf-8"))
    out = ROOT / "output" / "spec_to_pattern" / folder.name

    sp.dm.ensure_system_json()
    body_obj, body_yaml, lift = prepare_smplx_body("neutral")
    sp.BASE_BODY = BodyParameters(str(body_yaml))
    template = yaml.safe_load(sp.TEMPLATE.read_text(encoding="utf-8"))
    template = template.get("design", template)
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body_mesh = (bv * 100, bf)

    done = []
    for size in sizes:
        vals = info["sizes"][size]
        table = {sp.KEYS[c]: float(v) for c, v in zip(info["cols"], vals) if c in sp.KEYS and not isinstance(v, str)}
        target = {k: v for k, v in table.items() if k != "shoulder"}
        design, _, m, hist, params = sp.fit(target, template)
        name = f"spec_{folder.name.split('_')[0]}_{size}"
        print(f"[{size}] 版片 " + "  ".join(f"{NAMES[k]} {m[k]:.1f}/{table.get(k, '-')}" for k in NAMES) + "，縫合中…", flush=True)
        gfolder, x, f, _, _ = build(design, name, body_yaml, body_mesh)
        d = out / "尺碼" / size
        d.mkdir(parents=True, exist_ok=True)
        (d / "design.yaml").write_text(yaml.safe_dump(json.loads(json.dumps({"design": design}, default=float)), allow_unicode=True),
                                       encoding="utf-8")
        sp_obj = gfolder / f"{name}_sewn.obj"
        shutil.copy(sp_obj, d / f"{folder.name}_{size}.obj")  # 單位 cm、腳底在 y=0
        (d / "量測.json").write_text(json.dumps({"尺寸表": table, "版片量到": {k: round(v, 1) for k, v in m.items()},
                                               "參數": {k: round(v, 3) for k, v in params.items() if "肩寬" not in k},
                                               "頂點數": len(x)}, ensure_ascii=False, indent=1), encoding="utf-8")
        done.append((size, table, m, x, f))

    # 圖：商品照 ＋ 每個尺碼正面；最右邊三個尺碼疊在一起
    import matplotlib.image as mpimg
    keep = bv[bf].mean(1)[:, 1] * 100 > 60
    b = (body_mesh[0], bf[keep])
    n = len(done)
    fig, axes = plt.subplots(1, n + 2, figsize=(4.0 * (n + 2), 5.6))
    for ax in axes:
        ax.axis("off")
    img = folder / f"{folder.name}_front.jpg"
    if img.exists():
        axes[0].imshow(mpimg.imread(str(img)))
        axes[0].set_title("商品照", fontsize=13)
    for ax, (size, table, m, x, f) in zip(axes[1:], done):
        shade(ax, [(b[0], b[1], SKIN), (x, f, COLORS.get(size, "#5b8fd1"))])
        ax.set_xlim(-95, 95), ax.set_ylim(55, 175)
        ax.set_title(f"{size} 號：" + "／".join(f"{NAMES[k]} {m[k]:.1f}" for k in ["chest", "length", "sleeve"]), fontsize=11)
    ax = axes[-1]
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    # 疊圖：正面剪影，大的先畫、小的蓋在上面，露出來的邊就是尺碼差
    for size, table, m, x, f in sorted(done, key=lambda d: -d[2]["length"]):
        ax.add_collection(PolyCollection(x[f][:, :, :2], facecolors=COLORS.get(size, "#5b8fd1"), edgecolors="none"))
    ax.set_xlim(-95, 95), ax.set_ylim(55, 175)
    ax.set_aspect("equal")
    ax.legend(handles=[Patch(color=COLORS.get(s, "#5b8fd1"), label=s) for s, *_ in done], loc="lower center", ncol=n, fontsize=10)
    ax.set_title("疊在一起（露出來的邊＝尺碼差）", fontsize=12)
    fig.suptitle(f"{folder.name}：照尺寸表做 {'、'.join(sizes)}（胸寬、衣長、袖長照尺寸表；肩寬不照；單位 cm）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = out / f"尺碼_{'-'.join(sizes)}.png"
    fig.savefig(png, dpi=110)
    print("輸出：", png)


if __name__ == "__main__":
    main(*sys.argv[1:])
