# -*- coding: utf-8 -*-
"""不開瀏覽器，用 numpy 把「衣服＋貼圖（顏色、法線）＋立體零件＋身體」畫成圖（周誌、文件用）。
跟 output/*/viewer.html 一樣的東西：正交投影、z-buffer、顏色貼圖 × 光照，法線貼圖照切線空間（OpenGL +y）套上去，
強度 1.6（viewer 的 normalScale）。不是 Three.js 的畫面，光的設定比較簡單，但貼圖、零件位置是同一份檔案。
用法：見 fig_texture_journal.py。
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BODY = ROOT / "output" / "physics_export" / "_bodies" / "smplx_neutral_theta0_feet0.obj"
NORMAL_SCALE = 1.6


def read_obj(path):
    """回傳 v (n,3)、vt (m,2)、faces [(vi, ti)]、objects {名稱: 面的範圍}。"""
    v, vt, fv, ft, objs, cur, start = [], [], [], [], {}, None, 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            v.append([float(a) for a in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(a) for a in line.split()[1:3]])
        elif line.startswith("o ") or line.startswith("g "):
            if cur is not None:
                objs[cur] = (start, len(fv))
            cur, start = line.split()[1], len(fv)
        elif line.startswith("f "):
            ids = [a.split("/") for a in line.split()[1:]]
            for k in range(1, len(ids) - 1):  # 多邊形切成三角形
                tri = [ids[0], ids[k], ids[k + 1]]
                fv.append([int(a[0]) - 1 for a in tri])
                ft.append([int(a[1]) - 1 if len(a) > 1 and a[1] else -1 for a in tri])
    if cur is not None:
        objs[cur] = (start, len(fv))
    return np.array(v), (np.array(vt) if vt else None), np.array(fv), np.array(ft), objs


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def load_tex(path):
    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    return im


class Camera:
    """正交相機：繞直立軸轉 yaw（正值＝從人物左前方看）、往下 pitch 度；window＝世界座標 (x0, x1, y0, y1)（轉完以後的）。"""

    def __init__(self, yaw=0.0, pitch=0.0, window=(-0.4, 0.4, 0.9, 1.6), px_per_m=1400):
        y, p = np.radians(yaw), np.radians(pitch)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        self.R = Rx @ Ry  # 世界 → 相機（相機看 −z，+z 朝相機）
        self.view = self.R.T @ np.array([0, 0, 1.0])  # 世界座標中「朝相機」的方向
        self.win, self.s = window, px_per_m
        self.W = int((window[1] - window[0]) * px_per_m)
        self.H = int((window[3] - window[2]) * px_per_m)

    def project(self, p):
        q = p @ self.R.T
        return np.stack([(q[:, 0] - self.win[0]) * self.s, (self.win[3] - q[:, 1]) * self.s, q[:, 2]], axis=1)


class Scene:
    def __init__(self, cam, bg=(1, 1, 1)):
        self.cam = cam
        self.zbuf = np.full((cam.H, cam.W), -np.inf, np.float32)
        self.img = np.ones((cam.H, cam.W, 3), np.float32) * np.asarray(bg, np.float32)
        L = np.array([-0.45, 0.75, 1.0])
        self.L = L / np.linalg.norm(L)
        L2 = np.array([0.8, 0.2, 0.5])
        self.L2 = L2 / np.linalg.norm(L2)

    def _shade(self, n, base):
        """n (k,3) 已朝相機；base (k,3) 線性色。半球光＋主光＋補光。"""
        hemi = 0.42 + 0.18 * n[:, 1]
        d = 0.62 * np.clip(n @ self.L, 0, None) + 0.18 * np.clip(n @ self.L2, 0, None)
        return base * (hemi + d)[:, None]

    def add_mesh(self, v, f, color=None, vt=None, ft=None, tex=None, ntex=None, flat=False):
        """color：固定顏色 (3,)（sRGB）；或 tex（顏色貼圖）＋ ntex（法線貼圖）＋ vt/ft。"""
        cam = self.cam
        P = cam.project(v)
        vn = vertex_normals(v, f)
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        if tex is not None:  # 每個面的切線、副切線（UV 對位置的導數）
            dP1, dP2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
            duv1, duv2 = vt[ft[:, 1]] - vt[ft[:, 0]], vt[ft[:, 2]] - vt[ft[:, 0]]
            det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
            r = 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det)
            T = (dP1 * duv2[:, 1:2] - dP2 * duv1[:, 1:2]) * r[:, None]
            B = (dP2 * duv1[:, 0:1] - dP1 * duv2[:, 0:1]) * r[:, None]
            th, tw = tex.shape[:2]
        base_col = None if color is None else (np.asarray(color, np.float32) ** 2.2)
        for k, tri in enumerate(f):
            p = P[tri]
            x0, x1 = int(max(np.floor(p[:, 0].min()), 0)), int(min(np.ceil(p[:, 0].max()), cam.W - 1))
            y0, y1 = int(max(np.floor(p[:, 1].min()), 0)), int(min(np.ceil(p[:, 1].max()), cam.H - 1))
            if x0 > x1 or y0 > y1:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            a, b, c = p[0, :2], p[1, :2], p[2, :2]
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(den) < 1e-9:
                continue
            w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
            w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not inside.any():
                continue
            z = w0 * p[0, 2] + w1 * p[1, 2] + w2 * p[2, 2]
            iy, ix = np.nonzero(inside)
            iy, ix = iy + y0, ix + x0
            zz = z[inside]
            vis = zz > self.zbuf[iy, ix]
            if not vis.any():
                continue
            iy, ix, zz = iy[vis], ix[vis], zz[vis]
            W = np.stack([w0[inside][vis], w1[inside][vis], w2[inside][vis]], 1)
            N = fn[[k]].repeat(len(W), 0) if flat else W @ vn[tri]
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            flip = (N @ cam.view) < 0  # 雙面：背面翻過來
            N[flip] *= -1
            if tex is not None:
                uv = W @ vt[ft[k]]
                tx = np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)
                ty = np.clip(((1 - uv[:, 1]) * th).astype(int), 0, th - 1)
                base = tex[ty, tx] ** 2.2
                if ntex is not None:
                    nt = ntex[ty, tx] * 2 - 1
                    t = T[k] - N * (N @ T[k])[:, None]
                    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
                    bb = B[k] - N * (N @ B[k])[:, None] - t * (t @ B[k])[:, None]
                    bb /= np.maximum(np.linalg.norm(bb, axis=1, keepdims=True), 1e-12)
                    sgn = np.where(flip, -1.0, 1.0)[:, None]
                    n = t * (nt[:, 0:1] * NORMAL_SCALE) * sgn + bb * (nt[:, 1:2] * NORMAL_SCALE) * sgn + N * nt[:, 2:3]
                    N = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            else:
                base = np.repeat(base_col[None], len(W), 0)
            self.img[iy, ix] = self._shade(N, base)
            self.zbuf[iy, ix] = zz

    def add_body(self, color=(0.85, 0.76, 0.68)):
        v, _, f, _, _ = read_obj(BODY)
        self.add_mesh(v, f, color=color)

    def image(self):
        out = np.clip(self.img, 0, 1) ** (1 / 2.2)
        bg = np.isinf(self.zbuf)
        out[bg] = 1.0
        return (out * 255).astype(np.uint8)


def render(garment_dir, color_png, normal_png, cam, parts=False, tint=(1, 1, 1), part_colors=None, body=True):
    """garment_dir 裡的 garment_uv.obj ＋ 指定貼圖（＋ parts.obj）→ RGB 圖。tint：乘在顏色貼圖上（viewer 的材質顏色）。"""
    sc = Scene(cam)
    if body:
        sc.add_body()
    v, vt, f, ft, _ = read_obj(Path(garment_dir) / "garment_uv.obj")
    tex = load_tex(Path(garment_dir) / color_png) * np.asarray(tint, np.float32) if color_png else None
    ntex = load_tex(Path(garment_dir) / normal_png) if normal_png else None
    if tex is None:
        sc.add_mesh(v, f, color=tint)
    else:
        sc.add_mesh(v, f, vt=vt, ft=ft, tex=tex, ntex=ntex)
    if parts:
        pv, _, pf, _, objs = read_obj(Path(garment_dir) / "parts.obj")
        pc = part_colors or {"button": (0.93, 0.93, 0.90), "cord": (0.85, 0.85, 0.83), "aglet": (0.55, 0.55, 0.55)}
        for name, (a, b) in objs.items():
            kind = next((k for k in pc if name.startswith(k)), "button")
            sc.add_mesh(pv, pf[a:b], color=pc[kind], flat=kind == "button")
    return sc.image()
