# -*- coding: utf-8 -*-
"""
把 infer.py 的輸出（文字 JSON ＋ 76 個數值）轉成 GarmentCode 的 design.yaml，
再用指定身材生成網格並縫合（沿用 design_to_mesh.py、sew.py）。

文字部分決定「是哪一種」（上衣類型、領型、有沒有袖子…），
數值部分決定「多少」（衣長、寬鬆度、袖長…，0～1 正規化後還原到各參數的範圍）。
轉換邏輯直接用 ChatGarment 的 recursive_change_params_1float，和作者的流程一致。

要用 GarmentCode 的 .venv（不是 .venv-chatgarment）：
    .venv\\Scripts\\python tools\\chatgarment\\to_design.py <infer.py 的輸出資料夾> [--no-mesh]

需要 third_party/ChatGarment/assets 連到 GarmentCodeRC/assets（作者 README 的步驟 6）。
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CG_ROOT = REPO / "third_party" / "ChatGarment"
GC_ROOT = REPO / "third_party" / "GarmentCodeRC"


def load_chatgarment_utils():
    """只載入 garment_utils_v2.py 這一個檔案（不經過 llava 套件，避免需要 PyTorch）。
    它在匯入時用相對路徑讀 docs/ 和 assets/，所以要在 ChatGarment 根目錄匯入。"""
    os.chdir(CG_ROOT)
    sys.path.insert(0, str(GC_ROOT))
    spec = importlib.util.spec_from_file_location("garment_utils_v2", CG_ROOT / "llava" / "garment_utils_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_designs(gu, result_dir):
    """回傳 {'upper': design, ...}。上下身分開的輸出，76 個數值前後各 38 個；連身的整組 76 個。"""
    pred = json.loads((result_dir / "output.json").read_text(encoding="utf-8"))
    floats = np.load(result_dir / "float_preds.npy")
    if "upperbody_garment" in pred:
        floats = floats.reshape(2, -1)
        parts = {"upper": (pred["upperbody_garment"], floats[0]), "lower": (pred.get("lowerbody_garment"), floats[1])}
    else:
        parts = {"wholebody": (pred["wholebody_garment"], floats.reshape(-1))}

    designs = {}
    for name, (cfg, vals) in parts.items():
        if not cfg:
            continue
        float_dict = dict(zip(gu.all_float_paths, vals[: len(gu.all_float_paths)]))
        raw = cfg if "design" in cfg else {"design": cfg}
        design = gu.recursive_change_params_1float(copy.deepcopy(gu.designs_config), raw, float_dict,
                                                   invnorm_float=True, parent_path=None)["design"]
        designs[name] = design
    return designs


def summary(design):
    m, s, c, sl = design["meta"], design["shirt"], design["collar"], design["sleeve"]
    return (f"上衣={m['upper']['v']}、下身={m['bottom']['v']}、開襟={s['openfront']['v']}、"
            f"衣長={s['length']['v']:.2f}、寬鬆={s['width']['v']:.2f}、領型={c['f_collar']['v']}、"
            f"無袖={sl['sleeveless']['v']}、袖長={sl['length']['v']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=None,
                    help="infer.py 的輸出資料夾，預設是 output/chatgarment_ai 下最近更新的 4bit 或 bf16")
    ap.add_argument("--no-mesh", action="store_true", help="只產生 design.yaml，不生成網格")
    ap.add_argument("--body", default="mean_male")
    ap.add_argument("--tag", default=None, help="網格資料夾名稱的前綴，用來區分 4bit／bf16 的結果")
    args = ap.parse_args()
    if args.results is None:  # 接最近一次 infer.py 的結果（4bit 或 bf16）
        candidates = [p for p in (REPO / "output" / "chatgarment_ai" / m for m in ("4bit", "bf16")) if p.exists()]
        if not candidates:
            raise SystemExit("找不到 infer.py 的結果，請先跑「① AI 判斷款式」")
        args.results = max(candidates, key=lambda p: max(f.stat().st_mtime for f in p.rglob("meta.json")))
    results = Path(args.results).resolve()
    args.tag = args.tag or f"ai{results.name}"  # 4bit → ai4bit、bf16 → aibf16
    print(f"讀取：{results}（網格名稱前綴 {args.tag}）")

    gu = load_chatgarment_utils()
    todo = []
    for d in sorted(p for p in results.iterdir() if (p / "float_preds.npy").exists()):
        try:
            designs = to_designs(gu, d)
        except Exception as e:  # 模型輸出的 JSON 壞掉時記下來，不中斷其他張
            print(f"[{d.name}] 轉換失敗：{e}")
            continue
        for part, design in designs.items():
            (d / f"design_{part}.yaml").write_text(yaml.safe_dump({"design": design}, allow_unicode=True))
            print(f"[{d.name}] {part}：{summary(design)}")
            if part in ("upper", "wholebody") and design["meta"]["upper"]["v"]:
                todo.append((d, part, design))

    if args.no_mesh:
        return
    sys.path.insert(0, str(Path(__file__).parent))
    import design_to_mesh as dm  # 匯入時會切到 GarmentCodeRC 根目錄
    import sew

    dm.ensure_system_json()
    body = GC_ROOT / "assets" / "bodies" / args.body
    for d, part, design in todo:
        # 只取上衣：計畫書範圍是上半身，下身與腰帶關掉
        design = copy.deepcopy(design)
        design["meta"]["bottom"]["v"] = None
        design["meta"]["wb"]["v"] = None
        name = f"{args.tag}_{d.name}"
        if (dm.OUT_ROOT / name / f"{name}_sewn.obj").exists():
            print(f"[略過] {name}（已經縫好）")
            continue
        try:
            _, _, paths = dm.generate(design, body.with_suffix(".yaml"), name)
            sew.main(paths.g_box_mesh.parent, body.with_suffix(".obj"))
        except Exception as e:
            print(f"[{d.name}] 生成網格失敗：{e}")


if __name__ == "__main__":
    main()
