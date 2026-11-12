# -*- coding: utf-8 -*-
r"""試驗：照廠商尺寸表（平放量的 cm）直接打版，不靠 AI 的長度、寬度參數。

款式（圓領、長袖、袖口…）先借 i10 紅色圓領長袖 T 的 design_used.yaml，只改尺寸相關的參數：
  胸寬 → shirt.width     ＝ 胸寬 × 2 ÷ 身體胸圍（公式直接算，一次就準）
  衣長 → shirt.length    量後片「肩頸點到下襬」，按比例修正到準
  袖長 → sleeve.length   量袖片「肩點到袖口」那條邊，按比例修正到準
  肩寬 → 打版用的身體 shoulder_w
         Shirt 這種上衣的肩點跟著「身體肩寬」走，沒有自己的參數；
         所以只在打版用的尺寸檔裡改肩寬（等於把肩線往外移＝落肩），3D 身體不變。
全部只在 2D 版片上量，不縫合，每輪不到 1 秒。

跑法（在 newfashion_VT 根目錄）：
    .venv\Scripts\python tools\chatgarment\experiments\spec_to_pattern.py [網路衣服資料夾，預設 網路衣服_尺寸資料\z06_長袖T]
輸出：output/spec_to_pattern/<資料夾>/  每個尺碼的 design.yaml、版片圖，和 結果.json、版片比較.png
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from pygarment.pattern.core import BasicPattern  # noqa: E402

ROOT = dm.REPO
TEMPLATE = ROOT / "output" / "physics_export" / "neutral" / "i10_red_crew_long_sleeve_tee" / "design_used.yaml"
KEYS = {"身幅": "chest", "肩幅": "shoulder", "着丈": "length", "総丈": "length", "そで丈": "sleeve"}
TOL = 0.2  # cm
ROUNDS = 8


def sample_panel(panel, n=24):
    """版片外框取樣成折線（含曲線邊），回傳 (點, 每條邊的點列表)。"""
    v = np.asarray(panel["vertices"], float)
    edges = []
    for e in panel["edges"]:
        curve = BasicPattern._edge_as_curve(_P, v * [1, -1], e)  # GarmentCode 先把 y 翻成圖片座標再算曲線
        ts = np.linspace(0, 1, n)
        pts = np.array([[curve.point(t).real, -curve.point(t).imag] for t in ts])  # 翻回來
        edges.append(pts)
    return np.vstack(edges), edges


class _Flip:
    @staticmethod
    def _flip_y(p):
        return [p[0], -p[1]]


_P = _Flip()


def width_at(edges, y0):
    """版片外框在高度 y0 的水平寬度。"""
    xs = []
    for e in edges:
        for a, b in zip(e[:-1], e[1:]):
            if (a[1] - y0) * (b[1] - y0) <= 0 and a[1] != b[1]:
                xs.append(a[0] + (y0 - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
    return max(xs) - min(xs)


def measure(pattern):
    """在 2D 版片上量（cm）：胸寬（平放一半）、肩寬、衣長（後片肩頸點到下襬）、袖長（肩點到袖口）。"""
    panels = pattern["panels"]
    pts = {n: sample_panel(p)[0] for n, p in panels.items()}
    torso = [n for n in panels if "torso" in n]
    # 胸寬：在腋下（版片高度 55%，袖籠下方）橫切每片軀幹版片的寬度，加總÷2
    chest = sum(width_at(sample_panel(panels[n])[1], 0.55 * np.ptp(pts[n][:, 1]) + pts[n][:, 1].min())
                for n in torso) / 2
    back = pts["right_btorso"]
    length = np.ptp(back[:, 1])
    # 肩點：後片原始頂點中，在肩線上、離中心最遠的那個（袖籠曲線的上端）
    bv = np.asarray(panels["right_btorso"]["vertices"], float)
    top = bv[bv[:, 1] > bv[:, 1].max() - 8]
    shoulder = 2 * np.abs(top[:, 0]).max()
    # 袖長：袖片上沿（肩點到袖口的直邊）的長度
    sv = np.asarray(panels["right_sleeve_f"]["vertices"], float)
    sleeve = np.ptp(sv[:, 0])
    cuff = [n for n in panels if "cuff" in n and n.startswith("right")]
    if cuff:
        sleeve += min(np.ptp(np.asarray(panels[c]["vertices"])[:, 0]) for c in cuff[:1])
    return {"chest": chest, "shoulder": shoulder, "length": length, "sleeve": sleeve}


def build(design, body_params):
    body = copy.deepcopy(BASE_BODY)
    for k, v in body_params.items():
        body[k] = v  # __setitem__ 會重算 _shoulder_incl 等衍生量
    return MetaGarment("spec", body, design).assembly().pattern


def fit(target, template):
    design = copy.deepcopy(template)
    design["shirt"]["flare"]["v"] = 1.0
    bust = BASE_BODY.params["bust"]
    design["shirt"]["width"]["v"] = target["chest"] * 2 / bust
    shoulder_w = good_sw = BASE_BODY.params["shoulder_w"]
    history = []
    pat = m = None
    for r in range(ROUNDS + 6):
        try:
            new = build(design, {"shoulder_w": shoulder_w} if "shoulder" in target else {})
        except AssertionError:
            # 肩點推太外面：袖籠沒有空間，GarmentCode 畫不出來 → 退回一半
            history.append({"輪": r + 1, "失敗": f"打版用肩寬 {shoulder_w:.1f} 畫不出來（袖籠沒空間），退回"})
            shoulder_w = (shoulder_w + good_sw) / 2
            continue
        pat, good_sw = new, shoulder_w
        m = measure(pat)
        diff = {k: m[k] - target[k] for k in target}
        history.append({"輪": r + 1, "量到": {k: round(m[k], 1) for k in m}, "差": {k: round(d, 1) for k, d in diff.items()}})
        if all(abs(d) <= TOL for d in diff.values()) or len([h for h in history if "量到" in h]) >= ROUNDS:
            break
        if "length" in target:
            design["shirt"]["length"]["v"] *= target["length"] / m["length"]
        if "sleeve" in target:
            design["sleeve"]["length"]["v"] *= target["sleeve"] / m["sleeve"]
        if "shoulder" in target:
            shoulder_w -= diff["shoulder"]
    params = {"shirt.width": design["shirt"]["width"]["v"], "shirt.length": design["shirt"]["length"]["v"],
              "sleeve.length": design["sleeve"]["length"]["v"], "打版用肩寬 shoulder_w": good_sw}
    return design, pat, m, history, params


def draw(patterns, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    fig, axes = plt.subplots(1, len(patterns), figsize=(3.2 * len(patterns), 4.2), squeeze=False)
    for ax, (label, pat, m, tgt) in zip(axes[0], patterns):
        for name in ["right_btorso", "right_sleeve_f"]:
            _, es = sample_panel(pat["panels"][name])
            off = np.array([0, 0]) if "torso" in name else np.array([5, 50])
            for e in es:
                ax.plot(e[:, 0] + off[0], e[:, 1] + off[1], color="#2F7FC1" if "torso" in name else "#D9822B", lw=1.5)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-45, 75)
        ax.set_ylim(-5, 85)
        txt = "\n".join(f"{n}：{m[k]:.1f}（目標 {tgt[k]}）" for k, n in
                        [("chest", "胸寬"), ("shoulder", "肩寬"), ("length", "衣長"), ("sleeve", "袖長")] if k in tgt)
        ax.text(-43, 82, txt, fontsize=8, va="top")
        ax.tick_params(labelsize=7)
    fig.suptitle(title + "（藍＝後片右半、橘＝袖片，單位 cm）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main(folder=None):
    global BASE_BODY
    folder = Path(folder) if folder else ROOT / "網路衣服_尺寸資料" / "z06_長袖T"
    if not folder.is_absolute():
        folder = ROOT / folder
    info = json.loads((folder / "尺寸.json").read_text(encoding="utf-8"))
    dm.ensure_system_json()
    _, yml, _ = prepare_smplx_body("neutral")
    BASE_BODY = BodyParameters(str(yml))
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    template = template.get("design", template)

    out = ROOT / "output" / "spec_to_pattern" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    results, figs = {}, []
    for size, vals in info["sizes"].items():
        target = {KEYS[c]: float(v) for c, v in zip(info["cols"], vals) if c in KEYS and not isinstance(v, str)}
        design, pat, m, hist, params = fit(target, template)
        (out / f"design_{size}.yaml").write_text(yaml.safe_dump(json.loads(json.dumps({"design": design}, default=float)), allow_unicode=True), encoding="utf-8")
        results[size] = {"目標": target, "最後量到": {k: round(v, 1) for k, v in m.items()}, "參數": {k: round(v, 3) for k, v in params.items()},
                         "輪數": len(hist), "每輪": hist}
        figs.append((size, pat, m, target))
        print(f"[{size}] {len(hist)} 輪  " + "  ".join(f"{k} {m[k]:.1f}/{target[k]}" for k in target) +
              "  參數 " + ", ".join(f"{k}={v:.3f}" for k, v in params.items()), flush=True)
    (out / "結果.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    draw(figs, out / "版片比較.png", f"{info['name']}：照尺寸表打版")
    print("輸出：", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
