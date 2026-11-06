# -*- coding: utf-8 -*-
"""
第三批 50 件做成圖卡：每張圖 10 件（2 列 × 5 欄），每件是照片＋AI 判斷＋對錯。

對錯：
  領口：照 report_neckline.md（neck_report.py 產生）的「原始版型」「修正後」
  袖子：照 SLEEVE_TRUTH（看照片判斷的短袖／長袖／無袖），對 AI 描述和 sleeve 參數
  袖口：照片有羅紋縮口的（CUFF_TRUTH），看 cuff 參數有沒有選
照片是官方商品照，圖只限內部看，不上傳。

跑法（在 newfashion_VT 根目錄，先跑過 neck_report.py、to_design.py --no-mesh）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\batch_cards.py
輸出：output/chatgarment/第三批圖卡_1.png ～ _5.png、results/chatgarment_ai/bf16_i/judge_all.json
"""
import ast
import json
import sys
from pathlib import Path

import matplotlib
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "chatgarment_ai" / "bf16_i"
PHOTOS = ROOT / "output" / "chatgarment_ai" / "inputs_i"
OUT = ROOT / "output" / "chatgarment"
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 看照片判斷的袖子：短＝短袖、長＝長袖（i28 七分袖算長）、無＝無袖（2026-10-28 看圖卡填的）
LONG = ['i02', 'i04', 'i06', 'i08', 'i10', 'i12', 'i19', 'i23', 'i24', 'i28', 'i29', 'i30', 'i32', 'i33', 'i34', 'i36', 'i39', 'i43', 'i44', 'i45', 'i46', 'i47', 'i48', 'i49']
NONE = ['i25', 'i26']
SLEEVE_TRUTH = {f"i{k:02d}": "長" if f"i{k:02d}" in LONG else "無" if f"i{k:02d}" in NONE else "短" for k in range(1, 51)}
# 照片上袖口有一圈收口布的長袖：大學 T、毛衣、帽 T 的羅紋袖口，襯衫的扣子袖口（都可以用 CuffBand 近似）
CUFF_TRUTH = {"i02", "i06", "i08", "i19", "i23", "i32", "i34", "i36", "i43", "i44", "i45", "i46", "i47", "i48"}


def neck_rows():
    rows = {}
    for line in (RES / "report_neckline.md").read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.split("|")[1:-1]]
        if len(c) == 10 and c[0].startswith("i"):
            rows[c[0]] = {"描述": c[2], "原始": c[4], "原始對": c[5] == "✔", "修正後": c[6], "修正後對": c[7].startswith("✔")}
    return rows


def sleeve_kind(text, sleeveless, length):
    if sleeveless:
        return "無"
    return "長" if length >= 0.6 else "短"


def main():
    labels = json.loads((RES / "labels.json").read_text(encoding="utf-8"))
    necks = neck_rows()
    judge = {}
    for name, lab in labels.items():
        d = RES / name
        text = (d / "output.txt").read_text(encoding="utf-8")
        desc = ast.literal_eval(text.split("[描述]")[1].split("[版型]")[0].strip())
        part = "wholebody" if (d / "design_wholebody.yaml").exists() else "upper"
        g = desc.get("wholebody_garment" if part == "wholebody" else "upperbody_garment") or {}
        st = g.get("geometry_styles", {})
        sl = yaml.safe_load((d / f"design_{part}.yaml").read_text(encoding="utf-8"))["design"]["sleeve"]
        sdesc = ", ".join(st.get("sleeves") or [])
        made = sleeve_kind(sdesc, sl["sleeveless"]["v"], sl["length"]["v"])
        truth = SLEEVE_TRUTH.get(name[:3])
        cuff_desc = "cuff" in " ".join(st.get("extra") or []).lower()
        cuff_param = sl["cuff"]["type"]["v"]
        judge[name] = {
            "正確領型": lab["組別領型"], "AI 名稱": g.get("garment_name", "") + ("（連身）" if part == "wholebody" else ""),
            **necks[name],
            "袖子 照片": truth, "袖子 AI 描述": sdesc, "袖子 做出": made,
            "袖長參數": None if sl["sleeveless"]["v"] else round(float(sl["length"]["v"]), 2),
            "袖子對": None if truth is None else truth == made,
            "袖口 照片": name[:3] in CUFF_TRUTH, "袖口 描述": cuff_desc, "袖口 參數": cuff_param,
            "袖口對": (cuff_param is not None) == (name[:3] in CUFF_TRUTH),
        }
    (RES / "judge_all.json").write_text(json.dumps(judge, ensure_ascii=False, indent=2), encoding="utf-8")

    names = list(labels)
    mark = lambda ok: "（對）" if ok else ("（錯）" if ok is not None else "")
    for page in range(5):
        fig, axes = plt.subplots(2, 5, figsize=(22, 17))
        for ax, name in zip(axes.ravel(), names[page * 10:(page + 1) * 10]):
            j = judge[name]
            ax.imshow(Image.open(PHOTOS / f"{name}.png")); ax.axis("off")
            ax.set_title(f"{name[:3]}　正確：{j['正確領型']}　AI：{j['AI 名稱']}", fontsize=12)
            sleeve_line = f"袖子：照片 {j['袖子 照片'] or '？'}／做出 {j['袖子 做出']}" + \
                          (f"（{j['袖長參數']}）" if j["袖長參數"] is not None else "") + f"  {mark(j['袖子對'])}"
            cuff_line = f"縮口：照片 {'有' if j['袖口 照片'] else '無'}／描述 {'有' if j['袖口 描述'] else '無'}／參數 {j['袖口 參數'] or '無'}  {mark(j['袖口對'])}"
            txt = (f"AI 描述領子：{j['描述']}\n"
                   f"領口原始：{j['原始']}  {mark(j['原始對'])}\n"
                   f"領口修正後：{j['修正後']}  {mark(j['修正後對'])}\n"
                   f"{sleeve_line}\n{cuff_line}")
            ax.text(0.0, -0.03, txt, transform=ax.transAxes, va="top", fontsize=11.5, linespacing=1.5)
        fig.suptitle(f"第三批（bf16）{page * 10 + 1}～{page * 10 + 10}：AI 判斷與對錯", fontsize=17)
        plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=11)
        plt.savefig(OUT / f"第三批圖卡_{page + 1}.png", dpi=60)
        plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
