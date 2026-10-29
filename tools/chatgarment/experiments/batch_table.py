# -*- coding: utf-8 -*-
"""
一批 AI 推論結果的總表：每件一列，列出 AI 的描述與修正後的主要款式參數。

跑法（在 newfashion_VT 根目錄，先跑過 to_design.py --no-mesh）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\batch_table.py [results\\chatgarment_ai\\bf16_i]
輸出：<結果資料夾>/table_all.md
"""
import ast
import json
import sys
from pathlib import Path

import yaml

SHAPE = {"CircleNeckHalf": "圓", "CurvyNeckHalf": "圓(曲)", "CircleArcNeckHalf": "圓(弧)", "Bezier2NeckHalf": "圓(貝茲)",
         "VNeckHalf": "V", "SquareNeckHalf": "方", "TrapezoidNeckHalf": "梯形"}
COMP = {None: "－", "Turtle": "高領", "SimpleLapel": "翻領", "Hood2Panels": "帽子"}
UPPER = {"Shirt": "一般", "FittedShirt": "合身", None: "（無）"}


def main(results="results/chatgarment_ai/bf16_i"):
    root = Path(results)
    labels_path = root / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / "output.txt").exists()):
        text = (d / "output.txt").read_text(encoding="utf-8")
        desc = ast.literal_eval(text.split("[描述]")[1].split("[版型]")[0].strip())
        part = "wholebody" if (d / "design_wholebody.yaml").exists() else "upper"
        g = desc.get("wholebody_garment" if part == "wholebody" else "upperbody_garment") or {}
        st = g.get("geometry_styles", {})
        x = yaml.safe_load((d / f"design_{part}.yaml").read_text(encoding="utf-8"))["design"]
        s, c, sl = x["shirt"], x["collar"], x["sleeve"]
        extra = ", ".join(st.get("extra") or [])
        cuff_desc = "有" if "cuff" in extra.lower() else ""
        sleeve = "無袖" if sl["sleeveless"]["v"] else f"{sl['length']['v']:.2f}"
        lab = labels.get(d.name, {})
        rows.append(
            f"| {d.name} | {lab.get('組別領型', '')} | {g.get('garment_name', '')}{'（連身）' if part == 'wholebody' else ''} | "
            f"{UPPER.get(x['meta']['upper']['v'], x['meta']['upper']['v'])} | {'開' if s['openfront']['v'] else ''} | "
            f"{s['length']['v']:.2f} | {s['width']['v']:.2f} | "
            f"{', '.join(st.get('collar') or [])} | {SHAPE.get(c['f_collar']['v'], c['f_collar']['v'])}＋{COMP[c['component']['style']['v']]} | "
            f"{', '.join(st.get('sleeves') or [])} | {sleeve} | {cuff_desc} | {sl['cuff']['type']['v'] or '－'} |")
    head = [f"# {root.name} 總表（{len(rows)} 件）", "",
            "由 `tools/chatgarment/experiments/batch_table.py` 產生。參數是 `to_design.py` 修正後的（款式修正已套用，衣長、領深校正還沒套用）。", "",
            "- **衣長**：`shirt.length`（GarmentCode 預設 1.2）；**寬鬆**：`shirt.width`（1.0～1.3）；**袖長**：`sleeve.length`（1＝到手腕附近）",
            "- **描述有袖口**：AI 描述的 extra 有寫 cuffs（例如 ribbed cuffs）；**袖口參數**：`sleeve.cuff.type`", "",
            "| 衣服 | 正確領型 | AI 名稱 | 上衣 | 開襟 | 衣長 | 寬鬆 | AI 描述的領子 | 領型＋配件 | AI 描述的袖子 | 袖長 | 描述有袖口 | 袖口參數 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    (root / "table_all.md").write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    print(root / "table_all.md")


if __name__ == "__main__":
    main(*sys.argv[1:])
