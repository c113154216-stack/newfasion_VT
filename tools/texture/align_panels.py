# -*- coding: utf-8 -*-
"""
對齊方法 B：每片版片各自對齊，只用「縮放、平移、旋轉」，花紋不變形。

想法：版片的 UV 就是攤平的版片形狀（等比例縮小），平放的商品照也接近「版片攤開」。
  1. 每片版片的 UV → 正面（或背面）看的位置（公分）：用 box mesh 裡這片版片的擺法求一個相似變換
     （縮放＋旋轉＋平移，可含鏡射），版片形狀完全不變。
  2. 分成三組：身體（前後片）、左袖、右袖。
  3. 讓網格剪影和照片剪影重疊最多（IoU 最大）：
       a. 整件一起找「每公分幾個像素」與位置（全部版片共用同一個縮放 → 花紋大小一致）
       b. 每隻袖子再各自繞腋下轉個角度、微調位置（袖子平放的角度和 T-pose 不同）
       c. 全部一起微調
回傳每個三角形三個角在照片上的像素座標，給 bake_front.bake() 用。

跟 TPS（bake_front.align）的差別：TPS 會把網格彎過去完全貼合輪廓，邊緣準、但花紋可能被拉歪；
這個方法花紋不變形、尺寸正確，但網格和照片形狀不一樣的地方（例如 AI 估錯衣長）邊緣會對不滿。
"""
import cv2
import numpy as np
from scipy.optimize import minimize

WORK = 320  # 最佳化時把照片剪影縮到這個大小（長邊像素），算比較快


def similarity(src, dst):
    """2D 相似變換（Umeyama）：dst ≈ s·R·src + t，允許鏡射。回傳 3×2 矩陣 M，用法 [x, y, 1] @ M。"""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a)
    r = u @ vt  # 可能是鏡射（det = -1），UV 和 3D 擺法左右相反時需要
    s = sig.sum() / (a ** 2).sum()
    m = np.zeros((3, 2))
    m[:2] = (s * r).T
    m[2] = mu_d - s * mu_s @ r.T
    return m


def uv_to_cm(vt, faces, faces_uv, vertex_labels, box_v):
    """GarmentCode 的 UV 為了填滿正方形，x、y 是分開縮放的，版片在 UV 裡長寬比不對。
    用 box mesh 裡沒碰到縫線的邊（版片在 box mesh 裡是平的、長度是真的）反推 x、y 各自每單位幾公分：
    邊長² = ax²·du² + ay²·dv²，對 (ax², ay²) 做最小平方。"""
    stitch = np.array([lab.startswith("stitch") for lab in vertex_labels])
    rows, rhs = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ok = ~stitch[faces[:, a]] & ~stitch[faces[:, b]]
        d = vt[faces_uv[ok, a]] - vt[faces_uv[ok, b]]
        rows.append(d ** 2)
        rhs.append(((box_v[faces[ok, a]] - box_v[faces[ok, b]]) ** 2).sum(axis=1))
    (ax2, ay2), *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return vt * np.sqrt([ax2, ay2])


def panel_layout(box_v, vt, faces, faces_uv, panel, vertex_labels, view):
    """每個三角形三個角在「正面／背面看」的 2D 位置（公分，x 右、y 下），由各版片的 UV 經相似變換得到。
    縫線上的頂點前後片共用、box mesh 裡位置被拉扯過，求變換時不用它們。"""
    sign = 1.0 if view == "front" else -1.0  # 從背面看左右相反
    view_xy = np.column_stack([sign * box_v[:, 0], -box_v[:, 1]])
    vt = uv_to_cm(vt, faces, faces_uv, vertex_labels, box_v)  # 版片真實形狀（公分）
    corners = np.zeros((len(faces), 3, 2))
    for p in np.unique(panel):
        m = panel == p
        fv, fu = faces[m].ravel(), faces_uv[m].ravel()
        clean = np.array([not vertex_labels[i].startswith("stitch") for i in fv])
        if clean.sum() < 3:
            clean[:] = True
        _, keep = np.unique(fu[clean], return_index=True)
        mat = similarity(vt[fu[clean][keep]], view_xy[fv[clean][keep]])
        uv = vt[faces_uv[m]].reshape(-1, 2)
        corners[m] = (np.column_stack([uv, np.ones(len(uv))]) @ mat).reshape(-1, 3, 2)
    return corners


def groups_of(panel):
    """0＝身體，1＝穿的人的左袖，2＝右袖。"""
    g = np.zeros(len(panel), int)
    g[np.char.find(panel.astype(str), "left_sleeve") >= 0] = 1
    g[np.char.find(panel.astype(str), "right_sleeve") >= 0] = 2
    return g


def place(corners, group, pivots, params):
    """params = [s, tx, ty, θ1, dx1, dy1, θ2, dx2, dy2]：袖子先繞腋下轉、平移（公分），再整件縮放、平移到像素。"""
    s, tx, ty = params[:3]
    out = corners.copy()
    for g in (1, 2):
        th, dx, dy = params[3 * g: 3 * g + 3]
        c, sn = np.cos(th), np.sin(th)
        rot = np.array([[c, -sn], [sn, c]])
        m = group == g
        out[m] = (corners[m] - pivots[g]) @ rot.T + pivots[g] + [dx, dy]
    return out * s + [tx, ty]


def raster(tris, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, np.round(tris).astype(np.int32), 255)
    return mask


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum())


def align_panels(box_v, vt, faces, faces_uv, panel, vertex_labels, mask_p, view="front"):
    """回傳 (每個三角形三個角在照片上的像素座標 (F,3,2), 資訊 dict)。"""
    corners = panel_layout(box_v, vt, faces, faces_uv, panel, vertex_labels, view)
    group = groups_of(panel)

    # 腋下樞紐：袖子最靠近身體那一端的中心
    pivots = {}
    for g in (1, 2):
        pts = corners[group == g].reshape(-1, 2)
        inner = np.abs(pts[:, 0] - corners[group == 0].reshape(-1, 2)[:, 0].mean())
        pivots[g] = pts[inner < inner.min() + 2.0].mean(axis=0)

    # 縮小的照片剪影
    k = WORK / max(mask_p.shape)
    small = cv2.resize(mask_p, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    small = (small > 127).astype(np.uint8) * 255
    shape = small.shape

    def loss(params, free, sel=None, target=None):
        full = base.copy()
        full[free] = params
        tris = place(corners, group, pivots, full)
        if sel is None:
            return 1.0 - iou(raster(tris, shape), small)
        return 1.0 - iou(raster(tris[sel], shape), target)

    # 初始值：面積比求縮放、重心求位置
    area_cm = sum(abs(np.cross(t[1] - t[0], t[2] - t[0])) / 2 for t in corners[group == 0]) \
        + sum(abs(np.cross(t[1] - t[0], t[2] - t[0])) / 2 for t in corners[group > 0])
    ys, xs = np.nonzero(small)
    s0 = np.sqrt(len(xs) / max(area_cm / 2, 1e-6))  # 前後片重疊，面積大約算兩次
    c_cm = corners.reshape(-1, 2).mean(axis=0)
    base = np.array([s0, xs.mean() - s0 * c_cm[0], ys.mean() - s0 * c_cm[1], 0, 0, 0, 0, 0, 0], float)

    def run(free, step, sel=None, target=None):
        res = minimize(loss, base[free], args=(free, sel, target), method="Nelder-Mead",
                       options={"initial_simplex": np.vstack([base[free]] + [base[free] + np.eye(len(free))[i] * step[i]
                                                                           for i in range(len(free))]),
                                "xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
        base[free] = res.x
        return 1 - res.fun

    free_global = np.array([0, 1, 2])
    run(free_global, [0.1 * s0, 5, 5])                     # a. 整件
    # b. 兩隻袖子：目標是「身體沒蓋到的衣服範圍」，不然袖子會被拉去疊在身體上（那裡本來就是衣服，不扣分）
    torso = raster(place(corners, group, pivots, base)[group == 0], shape)
    rest = cv2.bitwise_and(small, cv2.bitwise_not(torso))
    for g in (1, 2):
        run(np.array([3 * g, 3 * g + 1, 3 * g + 2]), [0.25, 3, 3], group == g, rest)
    score = run(np.arange(9), [0.05 * s0, 3, 3, 0.1, 2, 2, 0.1, 2, 2])  # c. 全部微調

    params = base.copy()
    params[:3] = [params[0] / k, params[1] / k, params[2] / k]  # 換回原圖像素
    info = {"iou": float(score), "px_per_cm": float(params[0]),
            "sleeve_deg": [float(np.degrees(params[3])), float(np.degrees(params[6]))]}
    return place(corners, group, pivots, params), info
