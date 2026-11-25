"""① 程式對帳：不用量測程式，直接拿版片檔（specification.json）的頂點座標手算，和 量測.json 的「版片量到」比。"""
import json
from pathlib import Path
import numpy as np
R = Path(r"C:\Users\User\Desktop\newfashion_VT\output")
B = R / "spec_to_pattern/_batch"
rows = []
for g in sorted(B.glob("z0*")):
    for d in sorted(p for p in g.iterdir() if p.is_dir()):
        m = json.loads((d / "量測.json").read_text(encoding="utf-8"))
        name = f"specb_{g.name}_{d.name}"
        P = json.loads((R / "chatgarment" / name / f"{name}_specification.json").read_text(encoding="utf-8"))["pattern"]["panels"]
        V = lambda n: np.asarray(P[n]["vertices"], float)
        bv = V("right_btorso")
        # 肩寬：後片「肩線外端」頂點（最上面 8 cm 內離後中心最遠）× 2；後中心在 x=0
        top = bv[bv[:, 1] > bv[:, 1].max() - 8]
        shoulder = 2 * np.abs(top[:, 0]).max()
        length = np.ptp(bv[:, 1])
        sleeve = np.ptp(V("right_sleeve_f")[:, 0]) + sum(np.ptp(V(c)[:, 0]) for c in P if "cuff" in c and c.startswith(("right", "sl_right")) and c.endswith("_f"))
        # 胸寬：前後片在腋下（側縫上端）的寬度，各片頂點裡的 |x| 最大值加總（側縫是直線）÷2 ×2 片
        chest = sum(np.abs(V(n)[:, 0]).max() for n in P if "torso" in n) / 2
        hand = {"shoulder": shoulder, "length": length, "sleeve": sleeve, "chest": chest}
        got = m["版片量到"]
        rows.append((g.name, d.name, {k: (round(hand[k], 1), got[k], round(hand[k] - got[k], 1)) for k in hand}))
for gid, size, r in rows:
    print(f"{gid} {size:4s}", "  ".join(f"{k} 手算{v[0]} 程式{v[1]} 差{v[2]:+}" for k, v in r.items()))
