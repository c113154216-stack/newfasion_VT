# -*- coding: utf-8 -*-
"""
縫合參數實驗：同一件衣服用不同的縫合設定各縫一次，比較領口被拉長多少、衣服停在哪裡。

背景：縫合後的領口比 2D 版片長 35～38%（紅 T 71.6 → 99.1 cm）。
要先把縫合造成的拉長拿掉，才知道版片本身的領口大小對不對。

跑法（在 newfashion_VT 根目錄，先用 to_design.py 生成過網格）：
    .venv\\Scripts\\python tools\\chatgarment\\sew_experiment.py output\\chatgarment\\ai4bit_v2_g06_red_tee

輸出：同一資料夾的 *_sewn_<設定名稱>.obj，以及一張比較表（同時存成 sew_experiment.json）。
"""
import json
import pickle
import sys
import time
from pathlib import Path

import igl
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sew import mesh_edges, rest_lengths, sew, write_obj_with_new_positions  # noqa: E402

GC_BODIES = Path(__file__).resolve().parents[2] / "third_party" / "GarmentCodeRC" / "assets" / "bodies"

# 名稱 → sew() 的參數
VARIANTS = {
    "baseline": {},
    "limit5": {"stretch_limit": 0.05},
    "gravity": {"gravity": 0.02},
    "limit5_gravity": {"stretch_limit": 0.05, "gravity": 0.02},
    # 第二輪：邊更硬、重力更小
    "limit2_w100": {"stretch_limit": 0.02, "stretch_weight": 100.0},
    "limit2_w100_g002": {"stretch_limit": 0.02, "stretch_weight": 100.0, "gravity": 0.002},
    "limit2_w100_g005": {"stretch_limit": 0.02, "stretch_weight": 100.0, "gravity": 0.005},
    # 第三輪：分開看「領口」和「領口以外」各自讓衣服往上移多少（mask 在 main 裡換成實際的邊）
    "limit2_w100_neck_only": {"stretch_limit": 0.02, "stretch_weight": 100.0, "stretch_mask": "neck"},
    "limit2_w100_not_neck": {"stretch_limit": 0.02, "stretch_weight": 100.0, "stretch_mask": "not_neck"},
}


def openings(x, faces, edges):
    """把邊界邊串成一圈一圈的開口，回傳 [(開口內的邊在 edges 裡的索引, 平均高度)]，由高到低。
    OBJ 依 UV 把縫線上的頂點拆成好幾份，位置相同的要先當成同一點，否則每片版片會各自成一圈。"""
    fe = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1)
    u, c = np.unique(fe, axis=0, return_counts=True)
    boundary = u[c == 1]
    key = {tuple(e): k for k, e in enumerate(edges)}
    _, canon = np.unique(np.round(x, 4), axis=0, return_inverse=True)
    canon = canon.ravel()
    G = nx.Graph()
    for a, b in boundary:
        G.add_edge(canon[a], canon[b])
    loops = []
    for comp in nx.connected_components(G):
        ids = [key[tuple(e)] for e in boundary if canon[e[0]] in comp]
        loops.append((np.array(ids), x[edges[ids]].reshape(-1, 3)[:, 1].mean()))
    return sorted(loops, key=lambda t: -t[1])


def report(x, faces, edges, rest, body, neck_ids):
    length = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    ratio = length / rest
    bv, bf = body
    inside = igl.fast_winding_number_for_meshes(bv, bf, x) > 0.5
    return {
        "領口版片原長 cm": round(float(rest[neck_ids].sum()), 1),
        "領口縫合後 cm": round(float(length[neck_ids].sum()), 1),
        "領口拉長 %": round(float((length[neck_ids].sum() / rest[neck_ids].sum() - 1) * 100), 1),
        "所有邊：拉長超過 10% 的比例 %": round(float(np.mean(ratio > 1.1) * 100), 2),
        "所有邊：最大拉長 %": round(float((ratio.max() - 1) * 100), 1),
        "在身體裡的頂點 %": round(float(inside.mean() * 100), 2),
        "衣服最高點 y cm": round(float(x[:, 1].max()), 1),
        "下襬 y cm": round(float(x[:, 1].min()), 1),
        # 前中心線（x≈0、前面）的最高、最低點：看整件衣服往上下移了多少，以及布本身有沒有變長
        "前中心最高 y cm": round(float(x[(np.abs(x[:, 0]) < 1.5) & (x[:, 2] > 0), 1].max()), 1),
        "前中心最低 y cm": round(float(x[(np.abs(x[:, 0]) < 1.5) & (x[:, 2] > 0), 1].min()), 1),
    }


def main(folder, body_name="mean_male", only=None):
    folder = Path(folder)
    src = next(folder.glob("*_boxmesh.obj"))
    v, f = igl.read_triangle_mesh(str(src))
    orig_lens = pickle.loads(next(folder.glob("*_orig_lens.pickle")).read_bytes())
    bv, bf = igl.read_triangle_mesh(str(GC_BODIES / f"{body_name}.obj"))
    body = (bv * 100, bf)  # 身體網格是公尺，衣服是公分

    edges = mesh_edges(f)
    rest = rest_lengths(v, edges, orig_lens)
    neck_ids = openings(v, f, edges)[0][0]  # box mesh 上最高的那圈就是領口

    out_json = folder / "sew_experiment.json"
    results = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    for name, params in VARIANTS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        kwargs = dict(params)
        if kwargs.get("stretch_mask") in ("neck", "not_neck"):
            neck_mask = np.zeros(len(edges), bool)
            neck_mask[neck_ids] = True
            kwargs["stretch_mask"] = neck_mask if kwargs["stretch_mask"] == "neck" else ~neck_mask
        x = sew(v, edges, rest, body=body, **kwargs)
        write_obj_with_new_positions(src, src.with_name(src.name.replace("_boxmesh.obj", f"_sewn_{name}.obj")), x)
        results[name] = {"參數": params, "秒": round(time.time() - t0), **report(x, f, edges, rest, body, neck_ids)}
        print(name, json.dumps(results[name], ensure_ascii=False))

    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = [k for k in next(iter(results.values())) if k != "參數"]
    print("\n| 項目 | " + " | ".join(results) + " |")
    print("|---|" + "---|" * len(results))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(r[k]) for r in results.values()) + " |")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["mean_male"]), only=sys.argv[3:] or None)
