# -*- coding: utf-8 -*-
"""
用 SAM 2（Segment Anything）切出商品照的「整件衣服」和「領口開口」兩個遮罩，給 bake_front.py 用。

為什麼：
  - 去背：GrabCut 在白 T 配淺色背景時容易切錯；SAM 給一個框就能切出整件衣服。
  - 前片領口線：平拍照的輪廓最上面是後領，前領在更下面，中間露出後片內側和領標。
    在領口開口裡點幾下，SAM 會切出開口的範圍，範圍的下緣就是前片領口線，形狀照照片走（不規則也沒關係）。

要用另外的環境（.venv-sam：PyTorch CPU＋ultralytics），不能跟 GarmentCode 的 .venv 混：
    .venv-sam\\Scripts\\python tools\\texture\\sam_masks.py 照片.jpg [輸出資料夾]

環境變數 SAM_MODEL：模型檔位置，預設 D:\\newfasion_VT_work\\repo_deps\\models\\sam2.1_t.pt
（不存在時 ultralytics 會自動下載 sam2.1_t.pt 到目前資料夾）

輸出（預設跟照片同一個資料夾，檔名接在照片名後面）：
    <照片>_garment.png   整件衣服（白＝衣服）
    <照片>_opening.png   領口開口（白＝開口，含領標；下緣是羅紋外側的縫線）
    <照片>_neck.json     前片羅紋寬度（像素）
    <照片>_sam.png       檢查圖：照片上疊兩個遮罩和提示點
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = os.environ.get("SAM_MODEL", str(Path("D:/newfasion_VT_work/repo_deps/models/sam2.1_t.pt")))


def largest_component(mask):
    import cv2
    n, comp = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask
    sizes = np.bincount(comp.ravel())[1:]
    return comp == 1 + int(np.argmax(sizes))


def predict(pr, shape, multimask=True, **prompts):
    """回傳 SAM 的候選遮罩與分數。multimask＝True 時對同一組提示給 3 個候選（細部、部位、整體）。"""
    ms, bx = pr.inference_features(pr.features, src_shape=shape, multimask_output=multimask, **prompts)
    return ms.cpu().numpy().astype(bool), bx[:, 4].cpu().numpy()


def garment_mask(pr, rgb):
    """整張照片內縮 2% 的框當提示，切出整件衣服。框沒有歧義，只要單一結果
    （要 3 個候選再取最大的，會選到連背景一起的那個）。"""
    from scipy import ndimage
    h, w = rgb.shape[:2]
    m = int(0.02 * min(h, w))
    masks, _ = predict(pr, (h, w), multimask=False, bboxes=[[m, m, w - m, h - m]])
    best = masks[0]
    return ndimage.binary_fill_holes(largest_component(best))


def opening_mask(pr, rgb, garment):
    """在領口開口裡放提示點，切出開口。
    開口在後領最上緣中點的正下方；中間常有領標，所以「要」的點放在領標左右兩側，
    「不要」的點放在前片胸口和兩側。SAM 會給 3 個候選：最小的常常只有領標、最大的是整件衣服，
    另外兩個一個只有開口、一個連前片的羅紋也包進去。前片領口線是羅紋內緣，所以取面積合理的最小那個。"""
    ys, xs = np.nonzero(garment)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h, cx = x1 - x0, y1 - y0, int((x0 + x1) / 2)
    top = int(ys[xs == cx].min()) if (xs == cx).any() else y0      # 後領最上緣（中線）
    pos = [[cx - int(0.045 * w), top + int(0.05 * h)], [cx + int(0.045 * w), top + int(0.05 * h)]]
    neg = [[cx, top + int(0.22 * h)], [cx - int(0.25 * w), top + int(0.15 * h)], [cx + int(0.25 * w), top + int(0.15 * h)]]
    masks, scores = predict(pr, rgb.shape[:2], points=[pos + neg], labels=[[1] * len(pos) + [0] * len(neg)])
    area = garment.sum()
    inside = lambda m: all(m[y, x] for x, y in pos)
    ok = [m for m in masks if 0.005 * area < m.sum() < 0.06 * area and inside(m)]
    best = min(ok, key=lambda m: m.sum()) if ok else masks[int(np.argmax(scores))]
    return largest_component(best), (pos, neg)


def rib_width(pr, rgb, garment, opening):
    """前片羅紋的寬度（像素）。開口遮罩的下緣其實是羅紋外側的縫線（羅紋跟開口內側同色，SAM 分不開），
    真正的前片領口線是羅紋內緣。羅紋本身 SAM 切得出來但上緣很破碎，所以只拿它量「寬度」：
    中段每一行的上下厚度取中位數。前片領口線＝開口下緣往內平移這個寬度（bake_front.neckline_from_opening）。"""
    ys, xs = np.nonzero(garment)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h, cx = x1 - x0, y1 - y0, int((x0 + x1) / 2)
    top = int(ys[xs == cx].min()) if (xs == cx).any() else y0
    col = np.nonzero(opening[:, cx])[0]
    if len(col) == 0:
        return None
    ob = int(col.max())                                          # 開口在中線的下緣（羅紋外側）
    pos = [[cx, ob - 6], [cx - int(0.06 * w), ob - 10], [cx + int(0.06 * w), ob - 10]]
    neg = [[cx - int(0.045 * w), top + int(0.05 * h)], [cx + int(0.045 * w), top + int(0.05 * h)], [cx, ob + 20]]
    masks, _ = predict(pr, rgb.shape[:2], points=[pos + neg], labels=[[1] * 3 + [0] * 3])
    area = garment.sum()
    ok = [m for m in masks if 0.002 * area < m.sum() < 0.03 * area and m[pos[0][1], pos[0][0]]]
    if not ok:
        return None
    rib = min(ok, key=lambda m: m.sum())
    rys, rxs = np.nonzero(rib)
    rcx, half = (rxs.min() + rxs.max()) / 2, (rxs.max() - rxs.min()) / 2
    th = [rib[:, x].sum() for x in range(int(rcx - 0.4 * half), int(rcx + 0.4 * half)) if rib[:, x].any()]
    return float(np.median(th)) if th else None


def main(photo, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics.models.sam import SAM2Predictor
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]

    photo = Path(photo)
    out = Path(out_dir) if out_dir else photo.parent
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(photo).convert("RGB"))
    model = MODEL if Path(MODEL).exists() else "sam2.1_t.pt"
    pr = SAM2Predictor(overrides=dict(model=model, task="segment", mode="predict", imgsz=1024, save=False, verbose=False))
    pr.set_image(np.ascontiguousarray(rgb[:, :, ::-1]))   # SAM 吃 BGR；影像只編碼一次，兩個遮罩共用
    g = garment_mask(pr, rgb)
    o, (pos, neg) = opening_mask(pr, rgb, g)
    Image.fromarray((g * 255).astype(np.uint8)).save(out / f"{photo.stem}_garment.png")
    Image.fromarray((o * 255).astype(np.uint8)).save(out / f"{photo.stem}_opening.png")
    rib = rib_width(pr, rgb, g, o)
    import json
    (out / f"{photo.stem}_neck.json").write_text(json.dumps({"rib_px": rib}), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb); axes[0].contour(g, [0.5], colors="lime"); axes[0].set_title("整件衣服")
    ys, xs = np.nonzero(o)
    pad = 40
    sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    axes[1].imshow(rgb[sl], extent=(sl[1].start, sl[1].stop, sl[0].stop, sl[0].start))
    axes[1].contour(np.arange(sl[1].start, sl[1].stop), np.arange(sl[0].start, sl[0].stop), o[sl], [0.5], colors="cyan")
    for p in pos:
        axes[1].plot(*p, "g+", ms=12, mew=2)
    for p in neg:
        axes[1].plot(*p, "rx", ms=10, mew=2)
    axes[1].set_xlim(sl[1].start, sl[1].stop); axes[1].set_ylim(sl[0].stop, sl[0].start)
    axes[1].set_title("領口開口（青＝開口邊界；綠＋要、紅×不要）")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(out / f"{photo.stem}_sam.png", dpi=90); plt.close()
    print(f"衣服 {int(g.sum())} 像素；開口 {int(o.sum())} 像素；羅紋寬 {rib} 像素 → {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
