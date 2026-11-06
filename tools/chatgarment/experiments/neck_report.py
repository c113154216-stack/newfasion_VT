# -*- coding: utf-8 -*-
"""
第三批（領口比對 50 件）的領口整理：AI 每一步判斷得對不對、修正後的結果。

每件看四個地方：
  ① 描述：AI 第一步文字描述的 collar／hood 有沒有寫對領型
  ② 原始版型：AI 第二步給的 GarmentCode 參數（前領形狀、領子配件）對不對
  ③ 修正後：to_design.py 以描述為準修正後對不對
  ④ 領深：AI 原本的前領深 → neckline.py 校正後（平均男性身材）
正確答案是 labels.json 的組別（照商品頁寫的領型分組）。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\neck_report.py [results\\chatgarment_ai\\bf16_i]
輸出：<結果資料夾>/report_neckline.md
"""
import contextlib
import copy
import io
import json
import sys
from collections import Counter
from pathlib import Path

CWD = Path.cwd()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design_to_mesh as dm  # noqa: E402  匯入時切到 GarmentCodeRC 根目錄
import to_design as td  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from neckline import _depths, fix_neck_depth  # noqa: E402

ROUND = set(td.ROUND_SHAPES)
# 每組的正確答案：(描述要出現的關鍵字, 前領形狀可接受的集合（None＝GarmentCode 沒有對應形狀，不算對錯）, 領子配件)
EXPECT = {
    "A": (("crew", "round"), ROUND, None),
    "B": (("scoop",), ROUND, None),
    "C": (("v-neck", "v neck"), {"VNeckHalf"}, None),
    "D": (("square",), {"SquareNeckHalf", "TrapezoidNeckHalf"}, None),
    "E": (("boat", "bateau"), None, None),
    "F": (("turtle", "mock", "high neck", "funnel"), ROUND, "Turtle"),
    "G": (("henley",), ROUND, None),
    "H": (("polo",), None, "SimpleLapel"),
    "I": (("shirt collar", "button-down", "button down", "point collar", "spread collar"), None, "SimpleLapel"),
    "J": (("hood",), None, "Hood2Panels"),
    "K": (("sweetheart", "off-shoulder", "off the shoulder", "off-the-shoulder"), None, None),
}
SHORT = {"CircleNeckHalf": "圓", "CurvyNeckHalf": "圓(曲)", "CircleArcNeckHalf": "圓(弧)", "Bezier2NeckHalf": "圓(貝茲)",
         "VNeckHalf": "V", "SquareNeckHalf": "方", "TrapezoidNeckHalf": "梯形",
         None: "無", "Turtle": "高領", "SimpleLapel": "翻領", "Hood2Panels": "帽子"}


def raw_designs(gu, d):
    """不套用任何修正的版型（AI 第二步原本給的）。"""
    saved = td.fix_collar, td.fix_neck_shape, td.fix_sleeveless
    td.fix_collar = td.fix_neck_shape = td.fix_sleeveless = lambda design, styles: None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return td.to_designs(gu, d)
    finally:
        td.fix_collar, td.fix_neck_shape, td.fix_sleeveless = saved


def fixed_designs(gu, d):
    with contextlib.redirect_stdout(io.StringIO()):
        return td.to_designs(gu, d)


def judge(group, design):
    _, shapes, comp = EXPECT[group]
    shape, have = design["collar"]["f_collar"]["v"], design["collar"]["component"]["style"]["v"]
    ok_comp = have == comp
    ok_shape = shapes is None or shape in shapes or comp in ("Hood2Panels",)
    return ok_comp and ok_shape


def depth(design, body):
    d = copy.deepcopy(design)
    d["meta"]["bottom"]["v"] = None
    d["meta"]["wb"]["v"] = None
    if not d["meta"]["upper"]["v"]:
        d["meta"]["upper"]["v"] = "Shirt"  # 連身款被判成沒有上衣時，借 Shirt 量領口
    try:
        before = _depths(d, body, MetaGarment)
        info = fix_neck_depth(d, body, MetaGarment)
    except Exception:
        return None, None
    return (before[0] if before else None), (info["實際深度 cm"][0] if info and info["實際深度 cm"] else None)


def main(results="results/chatgarment_ai/bf16_i"):
    root = (CWD / results).resolve()
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    gu = td.load_chatgarment_utils()
    dm.ensure_system_json()
    body = BodyParameters(str(dm.GC_ROOT / "assets" / "bodies" / "mean_male.yaml"))

    rows, stats = [], {g: Counter() for g in EXPECT}
    for name, lab in labels.items():
        d, g = root / name, lab["組別"]
        styles_all = td.read_description(d)
        part = "upper" if (d / "design_upper.yaml").exists() else "wholebody"
        key = "upperbody_garment" if part == "upper" else "wholebody_garment"
        styles = (styles_all.get(key) or {}).get("geometry_styles", {})
        desc = ", ".join((styles.get("collar") or []) + [h for h in (styles.get("hood") or []) if "no hood" not in h])
        raw, fixed = raw_designs(gu, d)[part], fixed_designs(gu, d)[part]
        ok_desc = any(k in desc.lower() for k in EXPECT[g][0])
        ok_raw, ok_fix = judge(g, raw), judge(g, fixed)
        before, after = depth(fixed, body)
        stats[g].update({"件數": 1, "描述對": ok_desc, "原始對": ok_raw, "修正後對": ok_fix})

        def cell(x):
            return f'{SHORT.get(x["collar"]["f_collar"]["v"], x["collar"]["f_collar"]["v"])}＋{SHORT[x["collar"]["component"]["style"]["v"]]}'
        fix_note = "" if cell(raw) == cell(fixed) else ("✔ 修好" if ok_fix and not ok_raw else ("✘ 改錯" if ok_raw and not ok_fix else "改了"))
        depth_cell = f"{before:.1f} → {after:.1f}" if before and after else "量不到"
        rows.append(f"| {name} | {g} {lab['組別領型']} | {desc or '（沒寫）'} | {'✔' if ok_desc else '✘'} | "
                    f"{cell(raw)} | {'✔' if ok_raw else '✘'} | {cell(fixed)} | {'✔' if ok_fix else '✘'} {fix_note} | "
                    f"{'連身' if part == 'wholebody' else ''} | {depth_cell} |")
        print(rows[-1], flush=True)

    out = ["# 第三批領口比對（bf16，50 件）", "",
           "由 `tools/chatgarment/experiments/neck_report.py` 產生。正確答案＝`labels.json` 的組別（照商品頁寫的領型）。", "",
           "- **描述**：AI 第一步文字描述的 collar／hood 有沒有寫出正確領型",
           "- **原始版型**：AI 第二步的 GarmentCode 參數（前領形狀＋領子配件）",
           "- **修正後**：`to_design.py` 以描述為準修正後",
           "- 船領、心形領、一字領 GarmentCode 沒有對應形狀，只看領子配件；Polo、襯衫、帽 T 主要看領子配件",
           "- **前領深**：AI 原本 → `neckline.py` 校正後（cm，平均男性身材）", "",
           "## 各組統計", "", "| 組 | 領型 | 件數 | 描述對 | 原始版型對 | 修正後對 |", "|---|---|---|---|---|---|"]
    names = {lab["組別"]: lab["組別領型"] for lab in labels.values()}
    tot = Counter()
    for g, c in stats.items():
        tot.update(c)
        out.append(f"| {g} | {names[g]} | {c['件數']} | {c['描述對']} | {c['原始對']} | {c['修正後對']} |")
    out.append(f"| | **合計** | **{tot['件數']}** | **{tot['描述對']}** | **{tot['原始對']}** | **{tot['修正後對']}** |")
    out += ["", "## 每件", "", "| 衣服 | 正確答案 | AI 描述 | | 原始版型 | | 修正後 | | 備註 | 前領深 cm |",
            "|---|---|---|---|---|---|---|---|---|---|"] + rows
    (root / "report_neckline.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(root / "report_neckline.md")


if __name__ == "__main__":
    main(*sys.argv[1:])
