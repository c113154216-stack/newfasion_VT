# -*- coding: utf-8 -*-
r"""示範：自訂一個超小的尺碼，照 spec_batch.py 原本的規則打版、縫合（衣服小就小，不因為身體大去改它）。

跑法（在 newfashion_VT 根目錄）：
    .venv\Scripts\python flows\size_chart\experiments\tiny_size_demo.py z07_U領長袖T 3XS 33 30 50 50
    參數：衣服代號、尺碼名、之後照那件 尺寸.json 的 cols 順序給數字（z07：身幅 肩幅 総丈 そで丈）
輸出：output/spec_to_pattern/_tiny_demo/<代號>/<尺碼>/（格式同 _batch，可以直接給 spec_batch_export.py）
縫合資料夾用 spectiny_ 開頭，不會蓋掉正式結果。
"""
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "spec_batch.py"
code = SRC.read_text(encoding="utf-8")
patches = [
    ('info = json.loads((DATA / gid / "尺寸.json").read_text(encoding="utf-8"))\n    sp.BASE_BODY',
     'info = json.loads((DATA / gid / "尺寸.json").read_text(encoding="utf-8"))\n    info["sizes"] = TINY\n    sp.BASE_BODY'),
    ('name = f"specb_{gid.split(\'_\')[0]}_{size}"', 'name = f"spectiny_{gid.split(\'_\')[0]}_{size}"'),
]
for old, new in patches:
    assert code.count(old) == 1, old
    code = code.replace(old, new)

mod = types.ModuleType("spec_batch_tiny")
mod.__file__ = str(SRC)
sys.path.insert(0, str(SRC.parent))
exec(compile(code, str(SRC), "exec"), mod.__dict__)
mod.OUT = mod.sp.ROOT / "output" / "spec_to_pattern" / "_tiny_demo"
gid, size, *vals = sys.argv[1:]
mod.TINY = {size: [float(v) for v in vals]}

mod.sp.dm.ensure_system_json()
body_obj, body_yaml, _ = mod.prepare_smplx_body("neutral")
mod.run(gid, body_obj, body_yaml)
