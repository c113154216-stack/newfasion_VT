# -*- coding: utf-8 -*-
"""
比較 4-bit 和 bf16（原版精度）兩次 ChatGarment 推論的結果，輸出 Markdown 報告。

跑法（GarmentCode 的 .venv，先跑過兩種模式的 infer.py 和 to_design.py）：
    .venv\\Scripts\\python tools\\chatgarment\\compare.py output\\chatgarment_ai

輸出：output/chatgarment_ai/report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

KEYS = [  # (顯示名稱, design 裡的路徑)
    ("上衣類型", "meta.upper"), ("下身", "meta.bottom"), ("開襟", "shirt.openfront"),
    ("衣長", "shirt.length"), ("寬鬆", "shirt.width"), ("下襬張開", "shirt.flare"),
    ("前領型", "collar.f_collar"), ("前領深", "collar.fc_depth"), ("領子配件（帽子／翻領／高領）", "collar.component.style"),
    ("無袖", "sleeve.sleeveless"), ("袖長", "sleeve.length"), ("袖口寬", "sleeve.end_width"),
]


def get(design, path):
    node = design
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node.get("v") if isinstance(node, dict) else node


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return "—" if v is None else str(v)


def load(mode_dir, name):
    d = mode_dir / name
    out = {"meta": None, "design": None}
    if (d / "meta.json").exists():
        out["meta"] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for part in ("upper", "wholebody"):
        f = d / f"design_{part}.yaml"
        if f.exists():
            out["design"] = yaml.safe_load(f.read_text())["design"]
            break
    out["floats"] = np.load(d / "float_preds.npy") if (d / "float_preds.npy").exists() else None
    return out


def main(root):
    root = Path(root)
    q, b = root / "4bit", root / "bf16"
    names = sorted(p.stem for p in (root / "inputs").iterdir())
    lines = ["# ChatGarment：4-bit vs bf16（原版精度）", "",
             "同一張照片分別用兩種精度推論。bf16 和作者的設定相同，當作標準答案。", ""]

    agree, total, float_diffs = 0, 0, []
    for name in names:
        a, c = load(q, name), load(b, name)
        lines += [f"## {name}", ""]
        for label, m in (("4-bit", a["meta"]), ("bf16", c["meta"])):
            if m:
                lines.append(f"- {label}：{m['seconds']} 秒，顯示記憶體峰值 {m['peak_gpu_gib']} GiB")
        if a["design"] is None or c["design"] is None:
            lines += ["", "（其中一種模式沒有結果）", ""]
            continue
        lines += ["", "| 參數 | 4-bit | bf16 | 一致 |", "|---|---|---|---|"]
        for label, path in KEYS:
            va, vc = get(a["design"], path), get(c["design"], path)
            same = (abs(va - vc) < 0.05) if isinstance(va, float) and isinstance(vc, float) else va == vc
            agree += same
            total += 1
            lines.append(f"| {label} | {fmt(va)} | {fmt(vc)} | {'✅' if same else '❌'} |")
        if a["floats"] is not None and c["floats"] is not None and a["floats"].shape == c["floats"].shape:
            float_diffs.append(np.abs(a["floats"] - c["floats"]).mean())
        lines.append("")

    summary = [f"**整體**：{total} 個參數比較中有 {agree} 個一致（{agree / max(total, 1) * 100:.0f}%）。"]
    if float_diffs:
        summary.append(f"76 個正規化數值（0～1）的平均絕對差：{np.mean(float_diffs):.3f}。")
    lines[4:4] = summary + [""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
    print("\n".join(summary))


if __name__ == "__main__":
    main(sys.argv[1])
