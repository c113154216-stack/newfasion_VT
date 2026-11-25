"""每條縫線兩邊的版片邊長（cm）：兩邊不一樣長，縫起來就會一邊被壓縮（吃縫）。"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\size_chart")
import spec_to_pattern as sp
R = Path(r"C:\Users\User\Desktop\newfashion_VT\output\chatgarment")
for item in sys.argv[1:]:
    gid, size = item.split('/')
    s = json.loads((R / f"specb_{gid}_{size}" / f"specb_{gid}_{size}_specification.json").read_text(encoding="utf-8"))
    P = s["pattern"]["panels"]
    def elen(pn, ei):
        _, edges = sp.sample_panel(P[pn], n=200)
        e = edges[ei]
        return float(np.linalg.norm(np.diff(e, axis=0), axis=1).sum())
    print(item)
    for i, st in enumerate(s["pattern"]["stitches"]):
        if len(st) != 2: continue
        a, b = st
        la, lb = elen(a["panel"], a["edge"]), elen(b["panel"], b["edge"])
        flag = "  <-- 差 %.1f cm" % (la - lb) if abs(la - lb) > 0.5 else ""
        print(f"  stitch_{i:<2d} {a['panel']}[{a['edge']}] {la:5.1f}  ↔  {b['panel']}[{b['edge']}] {lb:5.1f}{flag}")
