"""攤平總表.json → 每項「攤平 − 版片」的差，按衣服分組。"""
import json
from pathlib import Path
import numpy as np

rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(__file__).with_name("flat_out").glob("z0*_*.json"))]
K = ("chest", "length", "sleeve", "shoulder")
print("衣服 尺碼 | " + " | ".join(f"{k}: 表/版/攤平A/衣身B/穿著" for k in K) + " | 穿著伸縮 5/50/95%")
diffs = {k: [] for k in K}
for r in rows:
    t, v, a, b, w = r["尺寸表"], dict(r["版片量到"]), r["攤平量到"], r["只撫平衣身量到"], r["縫好量到(穿著沿布面)"]
    if not r["衣服"].startswith("z06"):
        v["sleeve"] += 5.4  # 2D 量測程式漏算的袖口（sl_*_cuff 片，沿手臂方向 5.4 cm）
    cells = []
    for k in K:
        cells.append(f"{t.get(k, '-')}/{v[k]:.1f}/{a[k]}/{b.get(k, '-')}/{w[k]:.1f}")
        diffs[k].append((r["衣服"][:3], r["尺碼"], a[k] - v[k], (b.get(k, np.nan) - v[k])))
    print(r["衣服"][:3], r["尺碼"], "|", " | ".join(cells), "|", r["穿著時布的伸縮(版片=1) 5%/50%/95%"])
print()
for k in K:
    d = np.array([x[2] for x in diffs[k]])
    print(f"{k}: 攤平A−版片  中位 {np.median(d):+.1f}  範圍 {d.min():+.1f}～{d.max():+.1f}")
    print("    ", " ".join(f"{g}{s}:{x:+.1f}" for g, s, x, _ in diffs[k]))
