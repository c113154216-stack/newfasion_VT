# -*- coding: utf-8 -*-
"""把一批商品照整理成 ChatGarment 的輸入：照實際內容改名、統一轉成 PNG。

- infer.py 只吃 jpg／png，webp 要轉
- 有透明背景的圖，透明處填一個和衣服顏色差很多的底色（淺衣服配深底、深衣服配淺底），去背才切得開
- 原檔不動
- 有背面照的批次：正面放 <輸出資料夾>，背面放 <輸出資料夾>_back（之後貼顏色用；不放在一起，
  否則 infer.py 會把背面也當成一件衣服跑）

跑法（在 newfashion_VT 根目錄）：
    .venv\\Scripts\\python tools\\chatgarment\\import_photos.py [批次]
批次：g（預設，C:\\Users\\User\\Desktop\\newcloth → inputs_v2）
     h（C:\\Users\\User\\Desktop\\newcloth2 → inputs_h、inputs_h_back；來源見 newcloth2/sources.md）
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "output" / "chatgarment_ai"

# 每一批：來源資料夾、輸出資料夾、原檔名 → 新名稱（編號_顏色_款式；編號字母各批不重複：e、x、g、h…）
BATCHES = {
    "g": {
        "src": Path(r"C:\Users\User\Desktop\newcloth"),
        "dst": "inputs_v2",
        "names": {
            "bigImageT.webp": "g01_navy_fish_print_tee",
            "BlackT.webp": "g02_black_tee",
            "Hoodie_1.jpg": "g03_blue_hoodie",
            "Hoodie_2.jpg": "g04_white_hoodie",
            "LongT.png": "g05_beige_longsleeve_tee",
            "RedT.jpg": "g06_red_tee",
            "stripeT.jpg": "g07_grey_stripe_tee",
            "whiteT.png": "g08_white_tee",
        },
    },
    # 第二批：容易出問題的款式，每件正面＋背面（docs/找圖需求_第二批衣服.md）
    "h": {
        "src": Path(r"C:\Users\User\Desktop\newcloth2"),
        "dst": "inputs_h",
        "back": "_back",  # 背面照的檔名結尾；正面是 _front
        "names": {
            "black_cropped_tee": "h01_black_cropped_tee",
            "navy_longline_tee": "h02_navy_longline_tee",
            "navy_vneck_tee": "h03_navy_vneck_tee",
            "navy_polo": "h04_navy_polo",
            "frenchblue_button_down_shirt": "h05_frenchblue_button_down_shirt",
            "black_tank_top": "h06_black_tank_top",
            "charcoal_zip_hoodie": "h07_charcoal_zip_hoodie",
            "frenchblue_cardigan": "h08_frenchblue_cardigan",
            "navy_turtleneck_sweater": "h09_navy_turtleneck_sweater",
            "red_drop_shoulder_tee": "h10_red_drop_shoulder_tee",
            "green_striped_long_sleeve_tee": "h11_green_striped_longsleeve_tee",
            "black_back_print_tee": "h12_black_back_print_tee",
        },
    },
}

# 第三批：領口比對 50 件，只有正面（docs/找圖需求_第三批_領口比對.md）。
# 原檔名開頭的字母是領型組別（A 圓領…K 其他），新名稱拿掉組別和 _front，組別記在 results/.../labels.json
BATCH_I = [
    "A_blue_crew_tee", "A_brown_crew_sweatshirt", "A_green_crew_tee", "A_green_ribbed_long_sleeve_tee",
    "A_maroon_rib_baby_tee", "A_navy_crew_sweatshirt", "A_navy_crew_tee", "A_olive_crew_long_sleeve_tee",
    "A_pink_crew_tee", "A_red_crew_long_sleeve_tee",
    "B_green_scoop_tee", "B_navy_scoop_long_sleeve_tee", "B_purple_scoop_tee", "B_red_scoop_tee",
    "B_royal_scoop_dolman_tee",
    "C_blue_deep_vneck_tee", "C_charcoal_vneck_tee", "C_green_vneck_tee", "C_grey_vneck_sweater",
    "C_navy_vneck_tee", "C_red_deep_vneck_tee", "C_red_vneck_tee", "C_wine_vneck_sweater",
    "D_black_square_neck_long_sleeve_top", "D_burgundy_square_neck_tank", "D_grey_square_neck_tank",
    "D_navy_square_neck_rib_tee",
    "E_blue_boat_neck_top", "E_green_boat_neck_sweater", "E_red_boat_neck_top", "E_teal_boat_neck_blouse",
    "F_green_turtleneck_sweater", "F_navy_mock_neck_tee", "F_navy_turtleneck_sweater", "F_olive_mock_neck_tee",
    "F_red_turtleneck_sweater",
    "G_grey_henley_tee", "G_indigo_henley_tee", "G_red_henley_tee",
    "H_charcoal_polo", "H_green_polo", "H_red_polo",
    "I_burgundy_dress_shirt", "I_green_button_down_shirt", "I_grey_oxford_shirt",
    "J_blue_hoodie", "J_green_hoodie", "J_red_hoodie",
    "K_gray_off_shoulder_top", "K_red_sweetheart_neck_top",
]
BATCHES["i"] = {
    "src": Path(r"C:\Users\User\Desktop\newcloth3"),
    "dst": "inputs_i",
    "names": {f"{s}_front": f"i{k:02d}_{s[2:]}" for k, s in enumerate(BATCH_I, 1)},
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


def find(src, stem):
    """stem 可以是完整檔名，或不含副檔名的名稱（任何圖片格式都可以）。"""
    if (src / stem).exists():
        return src / stem
    return next((p for p in src.glob(stem + ".*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}), None)


def convert(path, out):
    im = flatten(Image.open(path))
    im.save(out)
    print(f"{path.name:40s} → {out.name}  {im.size[0]}×{im.size[1]}")


def main(batch="g"):
    cfg = BATCHES[batch]
    src, dst = cfg["src"], INPUTS / cfg["dst"]
    dst.mkdir(parents=True, exist_ok=True)
    back_dst = INPUTS / (cfg["dst"] + "_back")
    for old, new in cfg["names"].items():
        if "back" in cfg:
            pairs = [(old + "_front", dst / f"{new}.png"), (old + cfg["back"], back_dst / f"{new}_back.png")]
            back_dst.mkdir(parents=True, exist_ok=True)
        else:
            pairs = [(old, dst / f"{new}.png")]
        for stem, out in pairs:
            path = find(src, stem)
            if path is None:
                print(f"✘ 找不到 {src / stem}")
                continue
            convert(path, out)
    print(dst)


if __name__ == "__main__":
    main(*sys.argv[1:2])
