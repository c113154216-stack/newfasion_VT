# -*- coding: utf-8 -*-
r"""在衣服的 UV 貼圖上畫領口細節，另外輸出法線貼圖（假凹凸）。網格完全不動。

兩種領口：
  一般領口（沒有領子配件）：沿領口一圈羅紋帶＋車縫線（crew）
  高領（GarmentCode 的 Turtle）：領子本身就是一圈布條，只在領子和衣身的交界畫接縫＋車縫；
    領子本身有兩種：turtle-same 跟衣身同布料、turtle-rib 整片羅紋（turtle）
要不要加（2026-11-14 和林宇鑫定的）：沒有特殊領型就加；帽 T、Polo／襯衫（翻領）不加；V 領還沒定（V_NECK_TRIM）。

做法（全部在 UV 圖上，單位換算成 cm）：
  1. 把每片版片在 UV 上的範圍畫成遮罩；把「線」（領口那一圈，或高領和衣身的接縫）畫到貼圖上，記下每一點沿線的弧長（cm）
  2. 每片版片各自算「離線多遠」（距離轉換）和最近那一點的弧長
  3. 照距離決定畫什麼：羅紋帶、接縫、虛線車縫；顏色貼圖用灰色，法線貼圖由高度圖的梯度算出來
領口在哪：GarmentCode 的 vertex_labels.yaml 標了 *_collar 的頂點，網格邊界圈中含最多這種頂點的那一圈。
高領的接縫：sim_segmentation.txt 裡兩側的面分屬 *collar_* 版片和 *torso 版片的邊。

跑法（GarmentCode 的 .venv 就可以，只需要 numpy、scipy、Pillow、PyYAML）：
    python tools\texture\neck_trim.py <縫好的衣服.obj> <vertex_labels.yaml> <輸出資料夾> [貼圖邊長，預設 2048] [樣式] [design_used.yaml]
樣式：single／inner（一般領口）、turtle-same／turtle-rib（高領）。給了 design_used.yaml 就照 trim_style() 自動決定。
輸出：<輸出資料夾>/trim_<樣式>_color.png、trim_<樣式>_normal.png、plain_color.png（沒有細節的對照）
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

BAND_CM = 2.0          # 羅紋帶寬
RIB_PITCH_CM = 0.30    # 羅紋一條的間距
# 車縫線離領口的距離（cm）。真的 T 恤：一條緊貼接縫在衣身那一側，有的再一條在羅紋帶裡面靠接縫一點
STYLES = {"single": (BAND_CM + 0.12,), "inner": (BAND_CM + 0.12, BAND_CM - 0.35)}
TURTLE_STITCH_CM = 0.12  # 高領：車縫緊貼接縫在衣身那一側
DASH_CM, GAP_CM = 0.30, 0.12                  # 車縫線一針的長度、間隔
THREAD_CM = 0.05
GRAY = np.array([0.62, 0.63, 0.64])
NO_TRIM_COMPONENTS = {"Hood2Panels": "帽 T", "SimpleLapel": "翻領（Polo、襯衫）"}
V_NECK_TRIM = True


def trim_style(design, style="single"):
    """依版型決定領口畫什麼。回傳 (樣式或 None, 理由)。design：design_used.yaml 的 design。"""
    collar = design["collar"]
    comp = collar["component"]["style"]["v"]
    if comp in NO_TRIM_COMPONENTS:
        return None, f"{comp}：{NO_TRIM_COMPONENTS[comp]}，不加"
    if comp == "Turtle":
        return (style if style.startswith("turtle") else "turtle-same"), "高領：交界畫接縫＋車縫"
    if collar["f_collar"]["v"] == "VNeckHalf" and not V_NECK_TRIM:
        return None, "V 領：V_NECK_TRIM 關著，不加"
    return (style if not style.startswith("turtle") else "single"), f"{collar['f_collar']['v']}、沒有領子配件：加羅紋帶"


def read_obj(path):
    v, vt, fv, ft = [], [], [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            v.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(t) for t in line.split()[1:3]])
        elif line.startswith("f "):
            corners = [c.split("/") for c in line.split()[1:]]
            for i in range(1, len(corners) - 1):
                tri = [corners[0], corners[i], corners[i + 1]]
                fv.append([int(c[0]) - 1 for c in tri])
                ft.append([int(c[1]) - 1 for c in tri])
    return np.array(v), np.array(vt), np.array(fv), np.array(ft)


def neckline_edges(fv, ft, collar):
    """領口：網格的邊界圈中，含最多 collar 頂點的那一圈（整圈都取，接縫處的頂點沒有標 collar 也要算進去）。
    回傳 [(頂點 i, 頂點 j, uv i, uv j)]。"""
    count = {}
    for f, t in zip(fv, ft):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((f[a], f[b])))
            count.setdefault(key, []).append((f[a], f[b], t[a], t[b]))
    border = [e[0] for e in count.values() if len(e) == 1]
    # 邊界邊依頂點相接分成好幾圈
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            a = parent[a]
        return a
    for e in border:
        parent[find(e[0])] = find(e[1])
    loops = {}
    for e in border:
        loops.setdefault(find(e[0]), []).append(e)
    return max(loops.values(), key=lambda es: sum(e[0] in collar for e in es))


def turtle_seam(fv, ft, seg):
    """高領和衣身的接縫：兩側的面分屬 collar_ 版片和 torso 版片的邊。
    回傳 [(頂點 i, 頂點 j, 衣身那側 uv i, uv j, 高領那側 uv i, uv j)]。seg：每個頂點的版片名稱。"""
    def panel(f):
        names = [seg[i] for i in f if not seg[i].startswith("stitch")]
        return names[0] if names else ""
    by_edge = {}
    for f, t in zip(fv, ft):
        name = panel(f)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            by_edge.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append((name, {f[a]: t[a], f[b]: t[b]}))
    out = []
    for (i, j), sides in by_edge.items():
        body = [uv for name, uv in sides if "torso" in name]
        band = [uv for name, uv in sides if "collar_" in name]
        if len(sides) == 2 and body and band:
            out.append((i, j, body[0][i], body[0][j], band[0][i], band[0][j]))
    return out


def order_loop(edges):
    """邊串成一圈（依 3D 頂點相接），回傳依序排好的邊（每條邊前兩個是頂點、後兩個是 uv）。"""
    by_v = {}
    for e in edges:
        by_v.setdefault(e[0], []).append(e)
        by_v.setdefault(e[1], []).append(e)
    left, out = set(range(len(edges))), []
    idx = {id(e): i for i, e in enumerate(edges)}
    while left:
        i = left.pop()
        e = edges[i]
        chain, cur = [e], e[1]
        while True:
            nxt = [x for x in by_v.get(cur, []) if idx[id(x)] in left]
            if not nxt:
                break
            x = nxt[0]
            left.discard(idx[id(x)])
            if x[0] != cur:  # 方向反了就翻過來
                x = (x[1], x[0], x[3], x[2])
            chain.append(x)
            cur = x[1]
        out.extend(chain)
    return out


def vertex_arc(edges, v):
    """沿線（依頂點相接）每個頂點的弧長（cm），和總長。"""
    arc, s0 = {}, 0.0
    for i, j, _, _ in order_loop([(e[0], e[1], 0, 0) for e in edges]):
        arc.setdefault(i, s0)
        s0 += np.linalg.norm(v[i] - v[j])
        arc.setdefault(j, s0)
    return arc, s0


class Canvas:
    """一張貼圖的共用資料：版片遮罩、每片的編號、cm／像素，以及畫線、算距離的工具。"""

    def __init__(self, v, vt, fv, ft, size):
        self.size, self.vt = size, vt
        mask_img = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(mask_img)
        for t in ft:
            d.polygon([tuple(p) for p in self.px(vt[t])], fill=255)
        self.mask = np.asarray(mask_img) > 0
        self.islands, self.n = ndimage.label(self.mask)
        e3 = np.linalg.norm(v[fv[:, 0]] - v[fv[:, 1]], axis=1)  # cm／像素：3D 邊長 ÷ UV 邊長（中位數）
        e2 = np.linalg.norm(self.px(vt[ft[:, 0]]) - self.px(vt[ft[:, 1]]), axis=1)
        self.cm_per_px = float(np.median(e3[e2 > 1e-6] / e2[e2 > 1e-6]))

    def px(self, uv):
        return np.stack([uv[..., 0] * self.size, (1 - uv[..., 1]) * self.size], -1)

    def draw_lines(self, segs):
        """segs：[(uv 索引 a, uv 索引 b, 弧長 a, 弧長 b)] → (線的像素, 每個像素的弧長)"""
        seed = np.zeros((self.size, self.size), bool)
        arc = np.zeros((self.size, self.size))
        for ta, tb, sa, sb in segs:
            a, b = self.px(self.vt[ta]), self.px(self.vt[tb])
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) * 2))):
                x, y = (a + (b - a) * t).astype(int)
                if 0 <= x < self.size and 0 <= y < self.size:
                    seed[y, x] = True
                    arc[y, x] = sa + t * (sb - sa)
        return seed, arc

    def field(self, seed, arc):
        """畫到線的每片版片各自算：離線多遠（cm）、最近那一點的弧長。回傳 (距離, 弧長, 這些版片的遮罩)。"""
        dist = np.full((self.size, self.size), np.inf)
        along = np.zeros((self.size, self.size))
        touched = np.zeros((self.size, self.size), bool)
        for k in range(1, self.n + 1):
            isl = self.islands == k
            if not (seed & isl).any():
                continue
            dt, (iy, ix) = ndimage.distance_transform_edt(~(seed & isl), return_indices=True)
            dist[isl] = dt[isl] * self.cm_per_px
            along[isl] = arc[iy, ix][isl]
            touched |= isl
        return dist, along, touched


def stitch_line(dist, along, at):
    """離線 at cm 的一條虛線車縫：(線的像素, 車縫把布壓下去的高度)。"""
    dashes = (along % (DASH_CM + GAP_CM)) < DASH_CM
    return (np.abs(dist - at) < THREAD_CM) & dashes, -0.35 * np.exp(-((dist - at) / 0.06) ** 2)


def save(canvas, h, shade, stitch, out, name):
    color = np.clip(GRAY[None, None] * shade[..., None], 0, 1)
    color[stitch] = [0.48, 0.49, 0.50]
    color[~canvas.mask] = GRAY
    h = h + 0.25 * stitch  # 線本身微微浮起
    gy, gx = np.gradient(ndimage.gaussian_filter(h, 0.6), canvas.cm_per_px)
    strength = 0.08
    nrm = np.dstack([-gx * strength, gy * strength, np.ones_like(h)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Image.fromarray((color * 255).astype(np.uint8)).save(out / f"trim_{name}_color.png")
    Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save(out / f"trim_{name}_normal.png")
    Image.fromarray(np.full((canvas.size, canvas.size, 3), (GRAY * 255).astype(np.uint8))).save(out / "plain_color.png")


def crew(canvas, v, fv, ft, collar, style, out):
    """一般領口：羅紋帶＋車縫。"""
    edges = order_loop(neckline_edges(fv, ft, collar))
    segs, s0 = [], 0.0
    for i, j, ti, tj in edges:
        seg = np.linalg.norm(v[i] - v[j])
        segs.append((ti, tj, s0, s0 + seg))
        s0 += seg
    dist, along, _ = canvas.field(*canvas.draw_lines(segs))
    size = canvas.size
    band = dist < BAND_CM
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * along / RIB_PITCH_CM)
    h = np.zeros((size, size))
    h[band] = 0.35 + 0.25 * rib[band]                                       # 羅紋帶稍微高一點，一條一條起伏
    h -= 0.5 * np.exp(-((dist - 0.08) / 0.10) ** 2)                         # 上緣對折的圓邊
    h -= 0.9 * np.exp(-((dist - BAND_CM) / 0.07) ** 2)                      # 接縫往下陷
    stitch = np.zeros((size, size), bool)
    for at in STYLES[style]:
        m, dh = stitch_line(dist, along, at)
        stitch |= m
        h += dh
    shade = np.ones((size, size))
    shade[band] = 0.93 + 0.05 * rib[band]
    shade -= 0.25 * np.exp(-((dist - BAND_CM) / 0.05) ** 2)
    save(canvas, h, shade, stitch, out, style)
    return f"領口 {len(edges)} 段、一圈 {s0:.1f} cm"


def turtle(canvas, v, fv, ft, seg, style, out):
    """高領：領子和衣身的交界畫接縫＋一條車縫（衣身那側緊貼接縫）；turtle-rib 時領子整片畫羅紋。"""
    edges = turtle_seam(fv, ft, seg)
    arc, s0 = vertex_arc(edges, v)
    d_body, a_body, _ = canvas.field(*canvas.draw_lines([(e[2], e[3], arc[e[0]], arc[e[1]]) for e in edges]))
    d_band, a_band, band_area = canvas.field(*canvas.draw_lines([(e[4], e[5], arc[e[0]], arc[e[1]]) for e in edges]))
    size = canvas.size
    h = np.zeros((size, size))
    shade = np.ones((size, size))
    for d in (d_body, d_band):  # 接縫兩側都往下陷、顏色深一點
        h -= 0.9 * np.exp(-(d / 0.07) ** 2)
        shade -= 0.22 * np.exp(-(d / 0.05) ** 2)
    stitch, dh = stitch_line(d_body, a_body, TURTLE_STITCH_CM)
    h += dh
    if style == "turtle-rib":  # 羅紋沿接縫方向一條一條，整片領子都是
        rib = 0.5 + 0.5 * np.cos(2 * np.pi * a_band / RIB_PITCH_CM)
        h[band_area] += 0.35 + 0.25 * rib[band_area]
        shade[band_area] *= 0.93 + 0.05 * rib[band_area]
    save(canvas, h, shade, stitch, out, style)
    return f"高領接縫 {len(edges)} 段、一圈 {s0:.1f} cm"


def main(obj, labels, out, size=2048, style="single", design=None):
    size = int(size)
    if design:
        d = yaml.safe_load(open(design, encoding="utf-8"))
        style, why = trim_style(d.get("design", d), style)
        print(f"領口判斷：{why}")
        if style is None:
            return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    v, vt, fv, ft = read_obj(obj)
    v = v * (100 if np.ptp(v[:, 1]) < 5 else 1)  # 公尺 → cm
    canvas = Canvas(v, vt, fv, ft, size)
    if style.startswith("turtle"):
        seg = Path(str(labels).replace("_vertex_labels.yaml", "_sim_segmentation.txt")).read_text().splitlines()
        info = turtle(canvas, v, fv, ft, seg, style, out)
    else:
        lab = yaml.safe_load(open(labels, encoding="utf-8"))
        collar = {i for k, ids in lab.items() if "collar" in k for i in ids}
        info = crew(canvas, v, fv, ft, collar, style, out)
    print(f"{info}；UV 每像素 {canvas.cm_per_px * 10:.2f} mm；版片 {canvas.n} 片 → {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
