# -*- coding: utf-8 -*-
r"""縫好的 3D 衣服上的量測點與「沿布面」的精確距離（spec_batch.py 用）。

肩點：後片袖籠縫線（袖片↔後身那條縫）在版片座標上最高的那個頂點，也就是肩線、前後袖籠、袖子上緣交會的那一點。
      直接用 GarmentCode 的縫線資料找，不靠 3D 位置猜（2026-11-24 前用「後片最外側」或「袖籠上離領口最近」，
      落肩款會抓到肩點下面 2～3 cm 的袖籠上）。
肩頸點：後片（含和領口、帽子縫在一起的點）在版片座標上最高的頂點。
距離：每條邊加 STEINER_K 個點、同一個三角形內所有點兩兩相連，再跑 Dijkstra。在平的版片上驗證誤差 < 0.5%（只會偏長）。
      libigl 的 heat_geodesic 在這種縫線網格上會偏短約 2 cm（58.3 量成 56.5），exact_geodesic 在這版回傳空陣列，所以不用。
"""
import json

import igl
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

STEINER_K = 5


def _pairs(stitches):
    """縫線清單中正常的兩片縫線（有的會多一個 'right_wrong' 之類的字串）。"""
    for i, st in enumerate(stitches):
        if len(st) >= 2 and isinstance(st[0], dict) and isinstance(st[1], dict):
            yield i, st[0]["panel"], st[1]["panel"]


def landmarks(folder, name):
    """縫合資料夾（GarmentCode 輸出）→ 量測點的頂點編號：{"肩點": {left, right}, "肩頸點": {left, right}}。
    boxmesh 和縫好的網格頂點順序相同，boxmesh 的 xy 就是版片座標（每片平移過）。"""
    bx, _ = igl.read_triangle_mesh(str(folder / f"{name}_boxmesh.obj"))
    seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
    stitches = json.loads((folder / f"{name}_specification.json").read_text(encoding="utf-8"))["pattern"]["stitches"]
    tags = [set(s.split(",")) for s in seg]
    out = {"肩點": {}, "肩頸點": {}}
    for side in ("left", "right"):
        back = f"{side}_btorso"
        armhole = {f"stitch_{i}" for i, a, b in _pairs(stitches)
                   if {a, b} in ({f"{side}_sleeve_b", back}, {f"{side}_sleeve_f", back})}
        cand = np.array([v for v, t in enumerate(tags) if t & armhole])
        out["肩點"][side] = int(cand[bx[cand, 1].argmax()])
        touch = {f"stitch_{i}" for i, a, b in _pairs(stitches) if back in (a, b)}
        cand = np.array([v for v, (s, t) in enumerate(zip(seg, tags)) if s == back or t & touch])
        out["肩頸點"][side] = int(cand[bx[cand, 1].argmax()])
    return out


def surface_dist(x, f, src, dst, face_mask=None, k=STEINER_K):
    """網格表面上 src 到 dst（頂點編號，可以是一串）的最短距離（cm），只走 face_mask 的面。"""
    x = np.asarray(x, float)
    f = np.asarray(f, np.int64)
    if face_mask is not None:
        f = f[face_mask]
    n = len(x)
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(3, -1).T
    t = np.arange(1, k + 1) / (k + 1)
    pts = np.vstack([x, (x[uniq[:, 0]][:, None] * (1 - t)[None, :, None]
                         + x[uniq[:, 1]][:, None] * t[None, :, None]).reshape(-1, 3)])
    on_edge = n + np.arange(len(uniq) * k).reshape(-1, k)
    fp = np.hstack([f, on_edge[inv[:, 0]], on_edge[inv[:, 1]], on_edge[inv[:, 2]]])
    i, j = np.triu_indices(fp.shape[1], 1)
    a, b = fp[:, i].ravel(), fp[:, j].ravel()
    w = np.linalg.norm(pts[a] - pts[b], axis=1)
    ok = w > 1e-9
    g = coo_matrix((w[ok], (a[ok], b[ok])), shape=(len(pts),) * 2).tocsr()
    d = dijkstra(g, directed=False, indices=src)
    return float(d[dst]) if np.ndim(dst) == 0 else d[np.asarray(dst)]


def region(f, seg, keyword):
    """碰到某類版片（名稱含 keyword）的面，加上縫線那一條（三個頂點都是縫線點的面）。"""
    seg = np.asarray(seg)
    own = np.char.find(seg, keyword) >= 0
    st = np.char.startswith(seg, "stitch")
    return np.any(own[f], axis=1) | np.all(st[f], axis=1)


def shoulder(x, f, seg, lm):
    """肩寬：左肩點 → 右肩點，沿後片布面的最短距離。"""
    return surface_dist(x, f, lm["肩點"]["left"], lm["肩點"]["right"], region(f, seg, "btorso"))


def length(x, f, seg, lm):
    """衣長：肩頸點 → 後片下襬最近的一點，沿後片布面（等於後片攤平後從肩頸點垂直量到下襬），左右平均。"""
    seg = np.asarray(seg)
    hem = min(igl.all_boundary_loop(f), key=lambda lp: x[lp, 1].mean())
    back = (np.char.find(seg, "btorso") >= 0) | (seg == "wb_back")  # 有下襬羅紋的，量到羅紋底
    st = np.char.startswith(seg, "stitch")
    hb = np.array([v for v in hem if back[v] or st[v]])
    # 開襟衫：下襬、前開口、領口是同一圈布邊（肩頸點也在上面），只取後片最低處往上 5 cm 內的點
    hb = hb[x[hb, 1] < x[back, 1].min() + 5.0]
    mask = region(f, seg, "btorso") | region(f, seg, "wb_back")
    return float(np.mean([surface_dist(x, f, h, hb, mask).min() for h in lm["肩頸點"].values()]))


def sleeve(x, f, seg, lm):
    """袖長：肩點 → 袖口邊（袖子＋袖口那一圈布邊）最近的一點，只走袖子和袖口，左右平均。
    版片上袖片上緣是從肩點到袖口的直線，最近的袖口點就是上緣的另一端，所以等於「沿袖子上緣量」。"""
    seg = np.asarray(seg)
    st = np.char.startswith(seg, "stitch")
    loops = igl.all_boundary_loop(f)
    out = []
    for side, c in lm["肩點"].items():
        arm = np.array([s.startswith((f"{side}_sleeve", f"sl_{side}")) for s in seg])
        mask = np.any(arm[f], axis=1) | np.all(st[f], axis=1)
        cand = [lp for lp in loops if np.mean(arm[lp] | st[lp]) > 0.8 and np.sign(x[lp, 0].mean()) == np.sign(x[c, 0])]
        if not cand:
            continue
        cuff = np.asarray(max(cand, key=lambda lp: abs(x[lp, 0].mean())))
        out.append(float(surface_dist(x, f, c, cuff, mask).min()))
    return float(np.mean(out)) if out else float("nan")


def shoulder_inset(x, f, seg, lm, folder, name, inset=3.0):
    """診斷用（不是正式量法）：兩個肩點各往後中心內縮 inset cm（照版片座標找後片頂點），量這兩點的布面距離，
    再加回版片上那兩段。和正式肩寬的差＝肩點附近被縫合模擬壓縮的量（四條縫線交會處，半徑約 2 cm 被壓到 0.7）。"""
    bx, _ = igl.read_triangle_mesh(str(folder / f"{name}_boxmesh.obj"))
    panels = json.loads((folder / f"{name}_specification.json").read_text(encoding="utf-8"))["pattern"]["panels"]
    seg = np.asarray(seg)
    pts, add = [], 0.0
    for side in ("left", "right"):
        pn = f"{side}_btorso"
        # 版片上的肩角（後片頂端最外側的頂點）。縫線上的點在 boxmesh 是幾片版片位置的平均，不能拿來當版片座標。
        v = np.asarray(panels[pn]["vertices"], float)
        top = v[v[:, 1] > v[:, 1].max() - 8]
        corner = top[np.abs(top[:, 0]).argmax()] + np.asarray(panels[pn]["translation"])[:2]
        own = np.flatnonzero(seg == pn)
        target = corner + [-np.sign(corner[0] - bx[own, 0].mean()) * inset, 0.0]
        j = own[np.linalg.norm(bx[own, :2] - target, axis=1).argmin()]
        pts.append(j)
        add += float(np.linalg.norm(bx[j, :2] - corner))
    return surface_dist(x, f, pts[0], pts[1], region(f, seg, "btorso")) + add
