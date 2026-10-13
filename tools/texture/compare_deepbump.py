# -*- coding: utf-8 -*-
"""
比較兩種「一張照片 → 法線貼圖」的做法：
  A. 我們的經驗做法（normal_map.py）：亮度當高度，再取斜率
  B. DeepBump（https://github.com/HugoTini/DeepBump）：用神經網路由灰階照片預測法線

DeepBump 是 GPL-3 授權，所以不 import 它的程式碼，而是用它附的 cli.py 另外執行，
避免我們的專案被 GPL 約束。需要先跑 tools/setup/setup.ps1（會抓 DeepBump 並裝 onnxruntime）。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\texture\\compare_deepbump.py 照片.jpg [--crop 左 上 右 下]

輸出（output/texture_demo/）：<名稱>_deepbump_normal.png、<名稱>_compare.png
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from normal_map import OUT, estimate_normal, shade  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[2]
DEEPBUMP = REPO / "third_party" / "DeepBump"
LIGHTS = [([-1, -0.3, 0.9], "光從左上"), ([1, 0.3, 0.9], "光從右下")]


def run_deepbump(color_png, out_png):
    """用 DeepBump 的指令列工具產生法線貼圖，回傳單位法線（H×W×3，0～1 讀進來再轉 −1～1）。"""
    t = time.time()
    subprocess.run([sys.executable, str(DEEPBUMP / "cli.py"), str(color_png), str(out_png), "color_to_normals"],
                   cwd=DEEPBUMP, check=True, capture_output=True)
    n = np.asarray(Image.open(out_png).convert("RGB")).astype(np.float32) / 255 * 2 - 1
    return n / np.linalg.norm(n, axis=2, keepdims=True), time.time() - t


def match_convention(ours, theirs):
    """兩邊的 x、y 方向慣例可能不同（例如 OpenGL 的 y 朝上、影像的 y 朝下），
    用相關係數判斷要不要翻轉，讓兩者用同一套座標打光比較。"""
    fixed = theirs.copy()
    for c in (0, 1):
        if np.corrcoef(ours[..., c].ravel(), theirs[..., c].ravel())[0, 1] < 0:
            fixed[..., c] *= -1
    return fixed


def match_strength(ours, theirs):
    """DeepBump 的凹凸幅度天生比較小。把它的傾斜量等比例放大到和我們的平均傾斜量相同，
    比較的就是「抓到的結構」而不是強度（Three.js 也是用 normalScale 調強度）。"""
    tilt = lambda n: np.linalg.norm(n[..., :2], axis=2).mean()  # noqa: E731
    scaled = theirs.copy()
    scaled[..., :2] *= tilt(ours) / max(tilt(theirs), 1e-6)
    return scaled / np.linalg.norm(scaled, axis=2, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--crop", type=float, nargs=4, default=[0.36, 0.40, 0.64, 0.68])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    if not (DEEPBUMP / "cli.py").exists():
        raise SystemExit(f"找不到 DeepBump：{DEEPBUMP}，請先執行 tools\\setup\\setup.ps1")

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    l, t, r, b = args.crop
    crop = img.crop((int(w * l), int(h * t), int(w * r), int(h * b))).resize((args.size, args.size), Image.LANCZOS)
    rgb = np.asarray(crop).astype(np.float32) / 255

    OUT.mkdir(parents=True, exist_ok=True)
    name = Path(args.image).stem
    color_png = OUT / f"{name}_color.png"
    crop.save(color_png)

    t0 = time.time()
    ours = estimate_normal(rgb)
    ours_s = time.time() - t0
    theirs_raw, theirs_s = run_deepbump(color_png, OUT / f"{name}_deepbump_normal.png")
    theirs = match_convention(ours, theirs_raw)

    agree = float(np.mean(np.sum(ours * theirs, axis=2)))  # 平均 cos 夾角，1 = 完全同方向
    print(f"我們的做法 {ours_s:.2f} 秒、DeepBump {theirs_s:.1f} 秒；兩者方向一致度（平均 cos）{agree:.3f}")

    base = rgb.reshape(-1, 3).mean(axis=0)
    rows = [("A. 我們的做法（亮度→高度→斜率）", ours), ("B. DeepBump（神經網路，原始強度）", theirs),
            ("C. DeepBump（強度調到與 A 相同）", match_strength(ours, theirs))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4.3 * len(rows)))
    for i, (label, n) in enumerate(rows):
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("原始照片（裁切）", fontsize=13)
        axes[i, 1].imshow((n * np.array([1, -1, 1]) + 1) / 2)  # 顯示成 OpenGL 慣例，和存檔一致
        axes[i, 1].set_title(f"{label}\n法線貼圖", fontsize=13)
        for j, (light, light_name) in enumerate(LIGHTS):
            axes[i, 2 + j].imshow(shade(n, base, light))
            axes[i, 2 + j].set_title(f"純色＋法線，{light_name}", fontsize=13)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(f"一張照片 → 法線貼圖：兩種做法比較（{name}）", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / f"{name}_compare.png"
    plt.savefig(path, dpi=85)
    print(path)


if __name__ == "__main__":
    main()
