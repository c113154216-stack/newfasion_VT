# -*- coding: utf-8 -*-
"""
衣服關鍵點：用 DeepFashion2「短袖上衣」訓練好的模型（OpenMMLab mmpose 的 SimpleBaseline：ResNet50＋3 層反卷積），
找出領口、肩、袖口、腋下、下襬等關鍵點。只用 PyTorch＋torchvision 自己搭結構載入權重，不必裝 mmpose。

權重（約 137 MB）：
    https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_shirt_256x192-21e1c5da_20221208.pth
放在環境變數 DF2_MODEL 指的位置（預設 D:\\newfasion_VT_work\\repo_deps\\models\\df2_short_sleeved_shirt_res50.pth）。

要用 .venv-sam（有 PyTorch）：
    .venv-sam\\Scripts\\python tools\\texture\\garment_keypoints.py 照片.jpg [遮罩.png]
輸出 <照片>_keypoints.json（[[x, y, 信心], …]）與 <照片>_keypoints.png（檢查圖）。
遮罩（白＝衣服）用來決定裁切範圍；沒給就用整張照片。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

MODEL = os.environ.get("DF2_MODEL", str(Path("D:/newfasion_VT_work/repo_deps/models/df2_short_sleeved_shirt_res50.pth")))
INPUT_W, INPUT_H = 192, 256          # 模型輸入（寬 × 高）
# 輸出有 294 個通道（DeepFashion2 全部 13 類的關鍵點），短袖上衣是第 1～25 個
SHORT_SLEEVE_TOP = slice(0, 25)
MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255


class SimpleBaseline(nn.Module):
    def __init__(self, n_keypoints):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        layers, cin = [], 2048
        for _ in range(3):
            layers += [nn.ConvTranspose2d(cin, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            cin = 256
        self.deconv = nn.Sequential(*layers)
        self.final = nn.Conv2d(256, n_keypoints, 1)

    def forward(self, x):
        return self.final(self.deconv(self.backbone(x)))


def load_model(path=MODEL):
    st = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    n = st["keypoint_head.final_layer.weight"].shape[0]
    model = SimpleBaseline(n)
    # mmpose 的 ResNet 命名（backbone.conv1、backbone.layer1…）對到 torchvision 的 Sequential 編號
    names = {"conv1": "0", "bn1": "1", "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}
    new = {}
    for k, v in st.items():
        if k.startswith("backbone."):
            head, rest = k[len("backbone."):].split(".", 1)
            new[f"backbone.{names[head]}.{rest}"] = v
        elif k.startswith("keypoint_head.deconv_layers."):
            new["deconv." + k[len("keypoint_head.deconv_layers."):]] = v
        elif k.startswith("keypoint_head.final_layer."):
            new["final." + k[len("keypoint_head.final_layer."):]] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    assert not [m for m in missing if "num_batches_tracked" not in m], missing
    return model.eval()


def crop_box(rgb, mask=None, pad=1.25):
    """裁切框：衣服範圍往外擴 pad 倍，再補成 3:4（模型輸入的寬高比）。回傳 (中心, 寬, 高)。"""
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        h, w = rgb.shape[:2]
        x0, x1, y0, y1 = 0, w - 1, 0, h - 1
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    w, h = (x1 - x0) * pad, (y1 - y0) * pad
    if w / h > INPUT_W / INPUT_H:
        h = w * INPUT_H / INPUT_W
    else:
        w = h * INPUT_W / INPUT_H
    return c, w, h


def predict(model, rgb, mask=None):
    """回傳 (K, 3)：每個關鍵點在原照片上的 (x, y, 信心)。"""
    import cv2
    c, w, h = crop_box(rgb, mask)
    src = np.float32([[c[0] - w / 2, c[1] - h / 2], [c[0] + w / 2, c[1] - h / 2], [c[0] - w / 2, c[1] + h / 2]])
    dst = np.float32([[0, 0], [INPUT_W, 0], [0, INPUT_H]])
    M = cv2.getAffineTransform(src, dst)
    patch = cv2.warpAffine(rgb, M, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    x = torch.from_numpy(((patch - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        hm = model(x)[0].numpy()[SHORT_SLEEVE_TOP]                 # (25, 64, 48)
    K, hh, ww = hm.shape
    out = []
    for k in range(K):
        i = int(hm[k].argmax()); py, px = divmod(i, ww)
        # 往次高的鄰居偏 1/4 格（mmpose 的標準解碼）
        dx = 0.25 * np.sign(hm[k, py, min(px + 1, ww - 1)] - hm[k, py, max(px - 1, 0)])
        dy = 0.25 * np.sign(hm[k, min(py + 1, hh - 1), px] - hm[k, max(py - 1, 0), px])
        u, v = (px + dx + 0.5) * INPUT_W / ww, (py + dy + 0.5) * INPUT_H / hh
        Minv = cv2.invertAffineTransform(M)
        X, Y = Minv @ np.array([u, v, 1.0])
        out.append([float(X), float(Y), float(hm[k].max())])
    return np.array(out)


def main(photo, mask_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    photo = Path(photo)
    rgb = np.asarray(Image.open(photo).convert("RGB"))
    mpath = Path(mask_path) if mask_path else photo.with_name(f"{photo.stem}_garment.png")
    mask = (np.asarray(Image.open(mpath).convert("L")) > 127) if mpath.exists() else None
    kp = predict(load_model(), rgb, mask)
    photo.with_name(f"{photo.stem}_keypoints.json").write_text(json.dumps(kp.round(2).tolist()), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    for i, (x, y, s) in enumerate(kp):
        ax.plot(x, y, "o", ms=5, color="cyan" if s > 0.3 else "orange", mec="black")
        ax.text(x + 4, y - 4, str(i + 1), fontsize=8, color="yellow", bbox=dict(fc="black", alpha=0.5, pad=0.5, lw=0))
    ax.axis("off"); plt.tight_layout(); plt.savefig(photo.with_name(f"{photo.stem}_keypoints.png"), dpi=90); plt.close()
    print(f"{photo.name}：{len(kp)} 個關鍵點，信心中位數 {np.median(kp[:, 2]):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
