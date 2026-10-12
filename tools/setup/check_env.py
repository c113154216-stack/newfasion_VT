# -*- coding: utf-8 -*-
"""
檢查衣物生成需要的環境是否都到位（對應第四週週報「所需工具」）。

用任一個 Python 執行都可以，它會分別用兩個環境的 Python 去測試：
    .venv\\Scripts\\python tools\\setup\\check_env.py

檢查項目：
  - .venv：GarmentCode 能不能匯入、libigl 版本是不是 2.5.1
  - .venv-chatgarment：PyTorch 能不能用顯示卡、4-bit 量化能不能跑、ChatGarment 模型程式能不能匯入、SMPL-X 能不能產生人體
  - 檔案：第三方原始碼、assets 連結、ChatGarment 權重、SMPL-X 權重
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV = REPO / ".venv" / "Scripts" / "python.exe"
VENV_AI = REPO / ".venv-chatgarment" / "Scripts" / "python.exe"

# 每一項：(說明, 用哪個環境, 要執行的 Python 程式碼)
CHECKS = [
    ("GarmentCode 套件版本", VENV, """
import numpy, igl, CGAL, trimesh, yaml
from importlib.metadata import version
assert version('libigl') == '2.5.1', '要用 libigl 2.5.1，目前是 ' + version('libigl')
print('numpy', numpy.__version__, '・libigl', version('libigl'), '・trimesh', trimesh.__version__)
"""),
    ("GarmentCode 能打版", VENV, """
import os, sys
os.chdir(r'{gc}'); sys.path.insert(0, r'{gc}')
from assets.bodies.body_params import BodyParameters
from assets.garment_programs.meta_garment import MetaGarment
print('匯入成功')
"""),
    ("PyTorch 與顯示卡", VENV_AI, """
import torch
assert torch.cuda.is_available(), '抓不到 NVIDIA 顯示卡'
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
(x @ x).sum().item()
print('torch', torch.__version__, '・', torch.cuda.get_device_name(0))
"""),
    ("4-bit 量化（bitsandbytes NF4）", VENV_AI, """
import torch, bitsandbytes as bnb
layer = bnb.nn.Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16, quant_type='nf4').cuda()
layer(torch.randn(2, 512, device='cuda', dtype=torch.bfloat16))
print('bitsandbytes', bnb.__version__)
"""),
    ("ChatGarment 模型程式", VENV_AI, """
import sys, transformers
sys.path.insert(0, r'{cg}')
from llava.model.language_model.llava_garment_float50 import GarmentGPTFloat50ForCausalLM
print('transformers', transformers.__version__)
"""),
    ("SMPL-X 人體", VENV_AI, """
import smplx, torch
m = smplx.create(r'{smplx_dir}', model_type='smplx', gender='neutral', use_pca=False, ext='npz')
v = m(betas=torch.zeros(1, 10)).vertices[0].detach().numpy()
print('頂點', len(v), '・身高 %.2f m' % (v[:, 1].max() - v[:, 1].min()))
"""),
]

FILES = [
    ("ChatGarment 原始碼", REPO / "third_party/ChatGarment/llava"),
    ("GarmentCodeRC 原始碼", REPO / "third_party/GarmentCodeRC/pygarment"),
    ("assets 連結", REPO / "third_party/ChatGarment/assets/bodies"),
    ("ChatGarment 權重",
     REPO / "third_party/ChatGarment/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin"),
    ("SMPL-X 權重", REPO / "legacy/geometry_physics/assets/smplx/SMPLX_NEUTRAL.npz"),
]


def main():
    paths = {
        "gc": REPO / "third_party/GarmentCodeRC",
        "cg": REPO / "third_party/ChatGarment",
        "smplx_dir": REPO / "legacy/geometry_physics/assets",
    }
    ok = True

    print("== 檔案 ==")
    for name, path in FILES:
        exists = path.exists()
        ok &= exists
        extra = f"（{path.stat().st_size / 2**30:.1f} GB）" if exists and path.is_file() and path.stat().st_size > 2**30 else ""
        print(f"  {'✔' if exists else '✘'} {name}{extra}" + ("" if exists else f"：缺少 {path}"))

    print("== 環境 ==")
    for name, python, code in CHECKS:
        if not python.exists():
            print(f"  ✘ {name}：找不到 {python.parent.parent.name}，請先執行 tools\\setup\\setup.ps1")
            ok = False
            continue
        res = subprocess.run([str(python), "-W", "ignore", "-c", code.format(**paths)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        if res.returncode == 0:
            print(f"  ✔ {name}：{res.stdout.strip().splitlines()[-1]}")
        else:
            ok = False
            print(f"  ✘ {name}：{(res.stderr.strip().splitlines() or ['未知錯誤'])[-1]}")

    print("\n全部通過" if ok else "\n有項目沒通過，見上方 ✘")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
