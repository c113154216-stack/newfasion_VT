# -*- coding: utf-8 -*-
"""
顏色貼圖第 1 步：把平拍商品照的「正面」貼到生成衣服的前片與袖子正面上。

流程：
  1. 去背：用照片邊緣的顏色當背景，切出衣服的範圍（灰／白等單純背景適用）
  2. 平放：取衣服網格中前片、袖子正面的三角形，從正面投影成 2D 形狀（像平拍照）
  3. 對齊：先對齊兩個形狀的外框，再沿輪廓等距取對應點，用薄板樣條（thin-plate spline）
     把網格彎過去貼合照片，得到每個頂點在照片上的位置
  4. 烘焙：依每個三角形的 UV，把照片的顏色畫進貼圖（texture baking）；
     還沒處理的地方（後片、袖子背面）先用衣服的主色填滿
貼圖畫在版片的 UV 上，縫合只移動頂點、不動 UV，所以縫好的衣服會自動帶著花樣。

跑法（在 newfashion_VT 根目錄，先跑過 ① ② 產生衣服網格）：
    .venv\\Scripts\\python tools\\texture\\bake_front.py [--mesh ai4bit_e01_red_tee] [--photo 照片]

輸出（output/texture_bake/<衣服>/）：
    texture.png        顏色貼圖
    <衣服>.obj、.mtl    縫好的衣服＋貼圖（Three.js、Blender 可直接開）
    align.png          對齊檢查圖：照片上疊網格位置
    preview.png        貼好的衣服（正面、背面）
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "chatgarment"))
from gallery import draw, shaded  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

MESHES = REPO / "output" / "chatgarment"
OUT = REPO / "output" / "texture_bake"
BODIES = REPO / "third_party" / "GarmentCodeRC" / "assets" / "bodies"
FRONT_PANELS = ("ftorso", "sleeve_f")  # 正面看得到的版片
TEX_SIZE = 1024


# ---------- 讀檔 ----------
def read_obj(path):
    """讀 OBJ 的頂點、UV、面（含每個角的 UV 編號）。"""
    v, vt, f, ft = [], [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            corners = [c.split("/") for c in line.split()[1:4]]
            f.append([int(c[0]) - 1 for c in corners])
            ft.append([int(c[1]) - 1 for c in corners])
    return np.array(v), np.array(vt), np.array(f), np.array(ft)


def face_panels(faces, seg_file):
    """每個三角形屬於哪片版片：取它的頂點中第一個不是縫線的標記。"""
    labels = Path(seg_file).read_text().splitlines()
    out = []
    for tri in faces:
        names = [labels[i] for i in tri if not labels[i].startswith("stitch")]
        out.append(names[0] if names else "seam")
    return np.array(out)


# ---------- 1. 去背 ----------
def photo_mask(rgb):
    """GrabCut 去背：照片外圈當背景、其餘當「可能是衣服」，讓它分別學前景和背景的顏色分布，
    漸層背景、白衣服配淺灰背景這種顏色接近的情況也能切開。只留最大的一塊並補洞。"""
    h, w = rgb.shape[:2]
    m = int(0.03 * min(h, w))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, (m, m, w - 2 * m, h - 2 * m), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp = cv2.connectedComponents(mask)
    if n > 1:
        sizes = np.bincount(comp.ravel())[1:]
        mask = (comp == 1 + np.argmax(sizes)).astype(np.uint8) * 255
    return ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255


# ---------- 2、3. 平放與對齊 ----------
def raster(points, faces, shape):
    mask = np.zeros(shape, np.uint8)
    for tri in faces:
        cv2.fillConvexPoly(mask, np.round(points[tri]).astype(np.int32), 255)
    return mask


def contour(mask, per_segment=48):
    """最外圈輪廓上的對應點。先找 5 個特徵點：領口（上方中間最低處）、左右袖口（最左、最右）、
    下襬左右兩角；每兩個特徵點之間再等距取點。這樣即使袖子角度不同（A-pose vs 平放），
    袖口也一定對到袖口、下襬對到下襬。方向統一，起點都是領口。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    if cv2.contourArea(c.astype(np.float32), oriented=True) < 0:
        c = c[::-1]
    x0, x1 = c[:, 0].min(), c[:, 0].max()
    y0, y1 = c[:, 1].min(), c[:, 1].max()
    cx = (x0 + x1) / 2
    upper = (np.abs(c[:, 0] - cx) < 0.08 * (x1 - x0)) & (c[:, 1] < y0 + 0.5 * (y1 - y0))
    start = np.where(upper)[0][np.argmax(c[upper, 1])]  # 領口最低點
    c = np.roll(c, -start, axis=0)
    nx, ny = (c[:, 0] - x0) / (x1 - x0), (c[:, 1] - y0) / (y1 - y0)  # 正規化到 0～1
    marks = {0, int(np.argmin(nx)), int(np.argmax(nx)),
             int(np.argmax(ny - nx)), int(np.argmax(ny + nx))}  # 袖口、下襬兩角
    marks |= armpits(c, nx, ny)
    marks = sorted(marks)
    marks.append(len(c))
    pts = []
    for a, b in zip(marks[:-1], marks[1:]):
        seg = np.vstack([c[a:b], c[b % len(c)]])
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        t = np.linspace(0, d[-1], per_segment, endpoint=False)
        pts.append(np.column_stack([np.interp(t, d, seg[:, 0]), np.interp(t, d, seg[:, 1])]))
    return np.vstack(pts)


def armpits(c, nx, ny):
    """腋下：輪廓往內凹最深的地方（凸包缺口），左右各取一個，排除上方中間的領口。"""
    hull = cv2.convexHull(c.astype(np.int32).reshape(-1, 1, 2), returnPoints=False)
    try:
        defects = cv2.convexityDefects(c.astype(np.int32).reshape(-1, 1, 2), np.sort(hull, axis=0))
    except cv2.error:
        return set()
    found = {}
    for _, _, far, depth in (defects[:, 0] if defects is not None else []):
        if abs(nx[far] - 0.5) < 0.15 or not 0.05 < ny[far] < 0.75:
            continue  # 中間是領口；太上或太下不是腋下
        side = "L" if nx[far] < 0.5 else "R"
        if depth > found.get(side, (0, -1))[1]:
            found[side] = (int(far), depth)
    return {i for i, _ in found.values()}


def align(xy, faces, mask_p):
    """xy：所有頂點的 2D 投影（x 右、y 上）。輪廓用全部三角形（前片＋後片）投影出來，
    和平拍照一樣會從領口露出後領。回傳每個頂點在照片上的像素座標。"""
    ys, xs = np.nonzero(mask_p)
    p0, p1 = np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()])
    g = xy * [1, -1]  # 影像座標 y 朝下
    g0, g1 = g.min(axis=0), g.max(axis=0)
    g = p0 + (g - g0) / (g1 - g0) * (p1 - p0)  # 外框對齊
    src = contour(raster(g, faces, mask_p.shape))
    dst = contour(mask_p)
    tps = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=5.0)
    return tps(g), src, dst


# ---------- 4. 烘焙 ----------
def bake(uv_px, photo_px, faces_uv, faces_v, photo, size):
    """在 UV 貼圖上逐三角形算每個像素對應到照片的位置，最後一次取樣。"""
    map_x = np.full((size, size), -1, np.float32)
    map_y = np.full((size, size), -1, np.float32)
    for tri_uv, tri_v in zip(faces_uv, faces_v):
        a, b, c = uv_px[tri_uv]
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        lo, hi = np.clip(lo, 0, size - 1), np.clip(hi, 0, size - 1)
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        p = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
        m = np.array([b - a, c - a]).T
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        w = np.linalg.solve(m, (p - a).T).T  # 重心座標的 (w1, w2)
        inside = (w[:, 0] >= -1e-3) & (w[:, 1] >= -1e-3) & (w.sum(axis=1) <= 1 + 1e-3)
        if not inside.any():
            continue
        w = w[inside]
        pa, pb, pc = photo_px[tri_v]
        q = pa + w[:, :1] * (pb - pa) + w[:, 1:] * (pc - pa)
        px, py = xx.ravel()[inside], yy.ravel()[inside]
        map_x[py, px], map_y[py, px] = q[:, 0], q[:, 1]
    covered = map_x >= 0
    tex = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return tex, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="ai4bit_e01_red_tee", help="output/chatgarment 底下的衣服資料夾")
    ap.add_argument("--photo", default=None, help="正面平拍照，預設依衣服名稱找 output/chatgarment_ai/inputs")
    ap.add_argument("--body-obj", default=None,
                    help="預覽用的身體網格（公尺、腳底在 0）。預設：T-pose 衣服用 SMPL 模板，其他用 GarmentCode 男性平均")
    args = ap.parse_args()

    folder = MESHES / args.mesh
    name = args.mesh
    photo_path = Path(args.photo) if args.photo else find_photo(name)
    photo = np.asarray(Image.open(photo_path).convert("RGB"))

    box_v, vt, faces, faces_uv = read_obj(folder / f"{name}_boxmesh.obj")
    sewn_v, _, _, _ = read_obj(folder / f"{name}_sewn.obj")
    panel = face_panels(faces, folder / f"{name}_sim_segmentation.txt")
    front = np.array([any(k in p for k in FRONT_PANELS) for p in panel])
    print(f"照片：{photo_path.name}；前片＋袖子正面 {front.sum()} 個三角形／共 {len(faces)} 個")

    # 1. 去背
    mask_p = photo_mask(photo)
    base = np.median(photo[mask_p > 0], axis=0)

    # 2、3. 平放＋對齊（輪廓用全部版片，貼圖只用前片）
    photo_px, src, dst = align(box_v[:, :2], faces, mask_p)

    # 4. 烘焙
    uv = vt / max(1.0, vt.max())
    uv_px = np.column_stack([uv[:, 0], 1 - uv[:, 1]]) * TEX_SIZE
    tex, covered = bake(uv_px, photo_px, faces_uv[front], faces[front], photo, TEX_SIZE)
    tex[~covered] = base
    # 版片邊緣往外補幾個像素，避免取樣時接縫露出底色
    band = cv2.dilate(covered.astype(np.uint8), np.ones((7, 7), np.uint8)) & (~covered).astype(np.uint8)
    tex = cv2.inpaint(tex, band * 255, 3, cv2.INPAINT_TELEA)

    dst_dir = OUT / name
    dst_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tex).save(dst_dir / "texture.png")
    obj_text = (folder / f"{name}_sewn.obj").read_text().splitlines()
    obj_text[0] = f"mtllib {name}.mtl"
    (dst_dir / f"{name}.obj").write_text("\n".join(obj_text) + "\n")
    shutil.copy(folder / f"{name}_material.mtl", dst_dir / f"{name}.mtl")
    mtl = (dst_dir / f"{name}.mtl").read_text().splitlines()
    (dst_dir / f"{name}.mtl").write_text("\n".join(
        "map_Kd texture.png" if line.startswith("map_Kd") else line for line in mtl) + "\n")

    # 對齊檢查圖
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(photo); axes[0].contour(mask_p, [128], colors="lime", linewidths=1.5)
    axes[0].set_title("① 去背（綠線＝切出的衣服範圍）", fontsize=13)
    axes[1].imshow(photo)
    axes[1].triplot(photo_px[:, 0], photo_px[:, 1], faces[front], color="cyan", linewidth=0.15)
    axes[1].plot(*dst[::6].T, "o", color="lime", ms=3); axes[1].plot(*src[::6].T, "o", color="red", ms=3)
    axes[1].set_title("②③ 對齊（青＝網格；紅→綠＝輪廓對應點）", fontsize=13)
    axes[2].imshow(tex); axes[2].set_title("④ 烘焙出的顏色貼圖（UV）", fontsize=13)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(dst_dir / "align.png", dpi=80); plt.close()

    # 貼好的衣服：每個三角形取 UV 中心的顏色
    centers = uv_px[faces_uv].mean(axis=1).astype(int).clip(0, TEX_SIZE - 1)
    colors = tex[centers[:, 1], centers[:, 0]] / 255.0
    # 注意：<名稱>_body_measurements.yaml 會被 box mesh 步驟換成預設身材，實際打版用的是這個
    bv, bf = read_body(args.body_obj, folder / "body_measurements.yaml")
    fig = plt.figure(figsize=(12, 6.5))
    for k, (azim, title) in enumerate([(90, "正面"), (-90, "背面（第 1 步先填主色）")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        draw(ax, [(bv, bf, shaded(bv, bf, np.array([0.80, 0.66, 0.54]))), (sewn_v, faces, shaded(sewn_v, faces, colors))])
        ax.set_xlim(-85, 85); ax.set_ylim(-85, 85); ax.set_zlim(40, 175)
        ax.view_init(8, azim); ax.set_box_aspect((1, 1, 1)); ax.axis("off"); ax.set_title(title, fontsize=14)
    plt.tight_layout(); plt.savefig(dst_dir / "preview.png", dpi=85); plt.close()
    print(dst_dir)


def find_photo(mesh_name):
    """由衣服名稱找照片：去掉 ai4bit_／aibf16_ 前綴和 _r2 這類解析度後綴。"""
    import re
    stem = re.sub(r"^ai(4bit|bf16)_", "", re.sub(r"_r[\d.]+$", "", mesh_name))
    return next((REPO / "output" / "chatgarment_ai" / "inputs").glob(stem + ".*"))


def read_body(body_obj, measurements):
    """預覽用身體（公分、只取膝蓋以上）。沒指定時看衣服打版用的手臂角度決定用哪個身體。"""
    import igl
    import yaml
    if body_obj is None:
        tpose = yaml.safe_load(Path(measurements).read_text())["body"]["arm_pose_angle"] == 0
        body_obj = (REPO / "output" / "physics_export" / "_bodies" / "tpose_smpl_female.obj"
                    if tpose else BODIES / "mean_male.obj")
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    bv = bv * 100
    return bv, bf[bv[bf].mean(axis=1)[:, 1] > 60]


if __name__ == "__main__":
    main()
