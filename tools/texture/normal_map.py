# -*- coding: utf-8 -*-
"""
由一張衣服照片估計法線貼圖（normal map），讓 Three.js 的材質有布料的凹凸感。

做法（近似，不是真的量出凹凸）：
  1. 照片轉灰階，假設「亮＝凸、暗＝凹」當作表面高度
  2. 減掉大範圍的光影，只留細小的紋理（針織、絨毛…）
  3. 由高度的斜率算出每個點的表面方向，存成法線貼圖（xyz → RGB）
存檔用 OpenGL 慣例（綠色＝y 朝上），和 Three.js 的 normalMap 預設、DeepBump 一致。
限制：顏色本身有深淺的布（例如黑白條紋）會被誤判成凹凸；照片裡的皺褶也會被當成紋理。

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\texture\\normal_map.py 照片.jpg [--crop 0.36 0.40 0.64 0.68] [--strength 6]

輸出（output/texture_demo/）：
    <名稱>_color.png    顏色貼圖（裁切後）
    <名稱>_normal.png   法線貼圖
    <名稱>_demo.png     對照圖：原圖、法線貼圖、純色平面在兩種光線下的效果
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[2] / "output" / "texture_demo"


def estimate_normal(rgb, strength=6.0, detail_sigma=12.0, smooth=0.8):
    """rgb：0～1 的影像。回傳每個點的單位法線（H×W×3），影像座標（y 朝下）。"""
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    height = gray - ndimage.gaussian_filter(gray, detail_sigma)  # 去掉大範圍光影
    height = ndimage.gaussian_filter(height, smooth)             # 去掉雜訊
    dx = ndimage.sobel(height, axis=1)
    dy = ndimage.sobel(height, axis=0)
    n = np.dstack([-dx * strength, -dy * strength, np.ones_like(height)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def shade(normal, color, light):
    """純色平面只靠法線貼圖打光，用來看凹凸效果。"""
    light = np.asarray(light, float)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normal @ light, 0, 1)
    return np.clip(color * (0.25 + 0.85 * diffuse[..., None]), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--crop", type=float, nargs=4, default=[0.36, 0.40, 0.64, 0.68],
                    help="裁切範圍（左 上 右 下，0～1 的比例），挑平整的布料區域")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strength", type=float, default=6.0, help="凹凸強度")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    l, t, r, b = args.crop
    crop = img.crop((int(w * l), int(h * t), int(w * r), int(h * b))).resize((args.size, args.size), Image.LANCZOS)
    rgb = np.asarray(crop).astype(np.float32) / 255

    normal = estimate_normal(rgb, args.strength)
    opengl = normal * np.array([1, -1, 1])  # 影像 y 朝下 → OpenGL y 朝上（Three.js 預設）
    normal_rgb = (opengl + 1) / 2  # 法線貼圖的標準存法：−1～1 → 0～1

    OUT.mkdir(parents=True, exist_ok=True)
    name = Path(args.image).stem
    crop.save(OUT / f"{name}_color.png")
    Image.fromarray((normal_rgb * 255).astype(np.uint8)).save(OUT / f"{name}_normal.png")

    base = rgb.reshape(-1, 3).mean(axis=0)
    panels = [(rgb, "① 原始照片（裁切）"), (normal_rgb, "② 法線貼圖"),
              (shade(normal, base, [-1, -0.3, 0.9]), "③ 純色＋法線，光從左上"),
              (shade(normal, base, [1, 0.3, 0.9]), "④ 純色＋法線，光從右下")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.suptitle("由一張照片估計的法線貼圖：③④ 沒有用到照片的顏色，凹凸感全部來自法線貼圖", fontsize=14)
    plt.tight_layout()
    path = OUT / f"{name}_demo.png"
    plt.savefig(path, dpi=90)
    print(path)


if __name__ == "__main__":
    main()
