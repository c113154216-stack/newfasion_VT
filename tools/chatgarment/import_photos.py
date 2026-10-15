# -*- coding: utf-8 -*-
"""把一批商品照整理成 ChatGarment 的輸入：照實際內容改名、統一轉成 PNG。

- infer.py 只吃 jpg／png，webp 要轉
- 有透明背景的圖，透明處填一個和衣服顏色差很多的底色（淺衣服配深底、深衣服配淺底），去背才切得開
- 原檔不動

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\import_photos.py [來源資料夾] [輸出資料夾]
預設：C:\\Users\\User\\Desktop\\newcloth → output/chatgarment_ai/inputs_v2
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

# 原檔名 → 新名稱（照片實際內容：編號_顏色_款式）
NAMES = {
    "bigImageT.webp": "g01_navy_fish_print_tee",
    "BlackT.webp": "g02_black_tee",
    "Hoodie_1.jpg": "g03_blue_hoodie",
    "Hoodie_2.jpg": "g04_white_hoodie",
    "LongT.png": "g05_beige_longsleeve_tee",
    "RedT.jpg": "g06_red_tee",
    "stripeT.jpg": "g07_grey_stripe_tee",
    "whiteT.png": "g08_white_tee",
}


def flatten(im):
    """透明背景填上對比色；沒有透明就直接轉 RGB。"""
    im = im.convert("RGBA")
    rgba = np.asarray(im).astype(float)
    alpha = rgba[..., 3:] / 255.0
    if alpha.min() == 1.0:
        return im.convert("RGB")
    opaque = rgba[..., :3][alpha[..., 0] > 0.5]
    bright = opaque.mean() > 128 if len(opaque) else True
    bg = np.full(3, 40.0 if bright else 230.0)
    return Image.fromarray((rgba[..., :3] * alpha + bg * (1 - alpha)).astype(np.uint8))


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\User\Desktop\newcloth")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "output" / "chatgarment_ai" / "inputs_v2"
    dst.mkdir(parents=True, exist_ok=True)
    for old, new in NAMES.items():
        path = src / old
        if not path.exists():
            print(f"✘ 找不到 {path}")
            continue
        im = flatten(Image.open(path))
        im.save(dst / f"{new}.png")
        print(f"{old:16s} → {new}.png  {im.size[0]}×{im.size[1]}")
    print(dst)


if __name__ == "__main__":
    main()
