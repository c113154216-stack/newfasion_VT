# -*- coding: utf-8 -*-
"""
連身款（wholebody）保留下半段的實驗。

to_design.py 不分情況把 meta.bottom 設成 None（只取上衣），但連身款的下半段（例如 Skirt2）
是同一件衣服的一部分（長大衣、洋裝），不是 AI 腦補的另一件。這支只生成一件、保留下半段，
跟原本「拿掉下半段」的網格比較。縫合用原本的設定，差別只在下半段。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\experiments\\wholebody_keep_lower.py ^
        output\\chatgarment_ai\\bf16\\e23_khaki_trench aibf16_e23_khaki_trench_full

輸出：output/chatgarment/<名稱>/<名稱>_sewn.obj
"""
import sys
from pathlib import Path

import yaml

CWD = Path.cwd()  # design_to_mesh 匯入時會切換目錄，先記下來
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design_to_mesh as dm  # noqa: E402
import sew  # noqa: E402


def main(result_dir, name, body_name="mean_male"):
    result_dir = CWD / result_dir
    design = yaml.safe_load((result_dir / "design_wholebody.yaml").read_text(encoding="utf-8"))["design"]
    print("meta：", {k: v["v"] for k, v in design["meta"].items()})
    dm.ensure_system_json()
    body = dm.GC_ROOT / "assets" / "bodies" / body_name
    _, _, paths = dm.generate(design, body.with_suffix(".yaml"), name)
    sew.main(paths.g_box_mesh.parent, body.with_suffix(".obj"))


if __name__ == "__main__":
    main(*sys.argv[1:])
