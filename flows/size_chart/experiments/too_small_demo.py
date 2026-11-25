# -*- coding: utf-8 -*-
r"""示範：衣服比身體小（尺寸表胸寬×2 < 身體胸圍）時，拿掉 spec_batch.py 的保護會怎樣。

拿掉的兩個保護：
  1. 修正時胸寬不縮（每次縫完胸寬目標設回尺寸表）
  2. 挑最好那次時不看胸寬
其他完全照 spec_batch.py 跑。輸出到 output/spec_to_pattern/_too_small_demo/，縫合資料夾用 specdemo_ 開頭，
不會蓋掉正式結果。
2026-11-24 起 spec_batch.py 拿掉了 3D 修正迴圈（縫一次就好），這兩個保護跟著消失，這個示範現在只會照尺寸表縫一次；
要重現當時的現象，用 11-24 以前的 spec_batch.py。

跑法（在 newfashion_VT 根目錄）：
    .venv\Scripts\python flows\size_chart\experiments\too_small_demo.py z07_U領長袖T M
"""
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "spec_batch.py"
code = SRC.read_text(encoding="utf-8")
patches = [
    ('too_small = "chest" in target and target["chest"] * 2 < bust', "too_small = False"),
    ('if "chest" in goal and target["chest"] * 2 < bust:', "if False:"),
    ('name = f"specb_{gid.split(\'_\')[0]}_{size}"', 'name = f"specdemo_{gid.split(\'_\')[0]}_{size}"'),
    ('for size, vals in info["sizes"].items():', 'for size, vals in [(s, v) for s, v in info["sizes"].items() if s in DEMO_SIZES]:'),
]
KEEP = "--keep" in sys.argv  # 保留保護（對照組），輸出到 _too_small_demo/protected/（libigl 寫不了中文路徑）
if KEEP:
    sys.argv.remove("--keep")
    patches = patches[2:]
for old, new in patches:
    if old.startswith(("too_small =", 'if "chest" in goal')) and old not in code:
        continue  # 修正迴圈（和這兩個保護）已在 11-24 拿掉
    assert code.count(old) == 1, old
    code = code.replace(old, new)

mod = types.ModuleType("spec_batch_demo")
mod.__file__ = str(SRC)
sys.path.insert(0, str(SRC.parent))
exec(compile(code, str(SRC), "exec"), mod.__dict__)
mod.OUT = mod.sp.ROOT / "output" / "spec_to_pattern" / "_too_small_demo" / ("protected" if KEEP else "")
mod.DEMO_SIZES = set(sys.argv[2:]) or {"M"}

mod.sp.dm.ensure_system_json()
body_obj, body_yaml, _ = mod.prepare_smplx_body("neutral")
print("身體胸圍（打版用）：", mod.BodyParameters(str(body_yaml)).params["bust"], flush=True)
mod.run(sys.argv[1], body_obj, body_yaml)
