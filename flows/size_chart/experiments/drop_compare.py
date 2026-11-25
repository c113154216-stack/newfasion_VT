# -*- coding: utf-8 -*-
r"""落肩版型對照表（2026-11-25）：舊做法（_batch，Shirt）vs 落肩版型 (a)、(b)。

(a)＝先用 Shirt、肩寬推不到尺寸表（差 > 0.5）才換落肩版型；(b)＝判定是落肩款就換。
_batch_drop_b 是 (b) 跑出來的；(a) 換版型的尺碼和 (b) 一模一樣（同一個 fit2(drop_class=True)），
沒換的尺碼和舊做法一樣（肩角壓縮 ≤ 0.5，自動再縫不會多縫），所以 (a) 直接從兩邊組出來。
跑法（repo 根目錄）：.venv\Scripts\python flows\size_chart\experiments\drop_compare.py
輸出：印 Markdown 表，並寫 output/spec_to_pattern/_batch_drop_b/對照表.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SP = ROOT / "output" / "spec_to_pattern"
OLD, NEW = SP / "_batch", SP / (sys.argv[1] if len(sys.argv) > 1 else "_batch_drop_b")  # 第 1 個參數：新結果資料夾名
ITEMS = {"z01": ["S", "M", "L", "XL"], "z02": ["S", "M", "L", "XL"], "z04": ["S", "M", "L"], "z06": ["XXS", "XS", "S", "M", "L"]}
KEYS = [("shoulder", "肩寬"), ("chest", "胸寬"), ("length", "衣長"), ("sleeve", "袖長")]


def load(base, g, s):
    p = base / g / s / "量測.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows = []
    for g, sizes in ITEMS.items():
        for s in sizes:
            old, new = load(OLD, g, s), load(NEW, g, s)
            if not (old and new):
                print(f"{g} {s} 缺資料")
                continue
            tab = old["尺寸表"]
            a_switch = old["版片量到"]["shoulder"] < tab["shoulder"] - 0.5
            a = new if a_switch else old
            rows.append({"衣服": g, "尺碼": s, "尺寸表": tab, "舊": old["縫好量到"], "舊版片": old["版片量到"],
                         "b": new["縫好量到"], "b版片": new["版片量到"], "b版型": new.get("上衣版型"),
                         "a換版型": a_switch, "a": a["縫好量到"],
                         "舊壓縮": old.get("肩角被壓(診斷)"), "b壓縮": new.get("肩角被壓(診斷)"),
                         "b再縫": new.get("再縫每次的肩角壓縮")})
    print("| 衣服 | 尺碼 | 表 肩 | 舊 3D | (a) 3D | (b) 版片 | (b) 3D | (b) 再縫壓縮 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        t = r["尺寸表"]["shoulder"]
        f = lambda v: f"{v:.1f}（{v - t:+.1f}）"
        print(f"| {r['衣服']} | {r['尺碼']} | {t} | {f(r['舊']['shoulder'])} | {f(r['a']['shoulder'])}{' 換' if r['a換版型'] else ''} | "
              f"{r['b版片']['shoulder']:.1f} | {f(r['b']['shoulder'])} | {r['b再縫']} |")
    print()
    print("| 衣服 | 尺碼 | " + " | ".join(f"{n} 舊／(b)／表" for _, n in KEYS[1:]) + " |")
    print("|---|---|" + "---|" * (len(KEYS) - 1))
    for r in rows:
        print(f"| {r['衣服']} | {r['尺碼']} | " + " | ".join(
            f"{r['舊'][k]:.1f}／{r['b'][k]:.1f}／{r['尺寸表'][k]}" for k, _ in KEYS[1:]) + " |")
    (NEW / "對照表.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
