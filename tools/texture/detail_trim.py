# -*- coding: utf-8 -*-
r"""版片上沒有的細節：扣子、門襟、口袋（A：畫在貼圖上）＋扣子、帽繩的立體零件（B：小網格，貼在布的三角形上）。

位置用「版片上的公分」定義（DETAILS）：u＝離前中心多遠、w＝離下襬多高（前片版片座標），跟打版一樣好調。
A：先跑 seam_trim 的縫線，再用 extra 鉤子加細節；每片版片的版片座標 ↔ UV 用仿射擬合（只用不在縫線上的頂點），
   每個像素換回版片座標，照解析式畫（圓形扣子、門襟線、口袋輪廓）。顏色直接寫真的顏色（布色、扣子色），
   檢視時材質顏色用白色。
B：同樣的版片座標 → 找 boxmesh（版片平面）上包住它的三角形和重心座標 → 縫好網格上的 3D 位置和法向。
   扣子：小圓柱網格；帽繩：從帽子開口往下，每 1 cm 貼回衣服表面再往外 0.5 cm 的管子。
   另外輸出 parts_attach.json（每個零件掛在哪個三角形、重心座標、離布多遠），給 engine 讓零件跟著布動。
跑法（GarmentCode 的 .venv）：
    python tools\texture\detail_trim.py <衣服代號 z01|z02> <縫好的網格.obj（cm）> <縫合資料夾> <輸出資料夾> [design.yaml]
輸出：detail_color.png、detail_normal.png（A，含縫線）、seam_color.png、seam_normal.png（B 用，只有縫線＋口袋）、
     parts.obj（B 的零件，公尺）、parts_attach.json、garment_uv.obj
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seam_trim as st  # noqa: E402

# 細節定義（cm；u＝離前中心、w＝離下襬；顏色 0～1）
DETAILS = {
    "z02": {  # 襯衫：前中心門襟＋扣子（前片沒有開口，GarmentCode 的 openfront＝False）
        "fabric": [0.45, 0.58, 0.78], "button": [0.93, 0.93, 0.90],
        "placket_half": 1.75, "placket_stitch": 0.6,
        "button_r": 0.55, "button_top": 2.0, "button_pitch": 9.0, "button_bottom": 9.0,
    },
    "z01": {  # 帽 T：袋鼠口袋＋帽繩（B）
        "fabric": [0.17, 0.17, 0.18], "cord": [0.85, 0.85, 0.83], "aglet": [0.55, 0.55, 0.55],
        "pocket": {"bottom": 6.0, "height": 19.0, "top_half": 11.0, "bottom_half": 17.0, "opening_to": 11.0},
        "cord_from_cf": 4.0, "cord_len": 32.0, "cord_r": 0.3,
    },
}


def panel_frames(v, vt, fv, ft, seg, stitches, canvas, box, spec):
    """每片前片版片：版片座標 ↔ UV 的仿射、前中心的 x、下襬的 y，和它在 UV 圖上的像素遮罩。"""
    labs = st.face_panels(fv, seg, stitches)
    out = {}
    for pn in ("left_ftorso", "right_ftorso"):
        fs = [k for k, l in enumerate(labs) if l == pn]
        if not fs:
            continue
        P, U = [], []
        for k in fs:
            for c in range(3):
                vi = fv[k][c]
                if not seg[vi].startswith("stitch"):
                    P.append(box[vi, :2]); U.append(vt[ft[k][c]])
        P, U = np.array(P), np.array(U)
        A = np.c_[P, np.ones(len(P))]
        M, *_ = np.linalg.lstsq(A, U, rcond=None)  # uv = [x y 1] @ M
        # 前中心、下襬、領口最低點直接從版片規格讀：前片半片的本地座標 x＝0 是前中心（GarmentCode 的 TorsoFrontHalfPanel），
        # 加上版片的平移就是 boxmesh 座標。不要從頂點猜：前中心上的頂點是縫線點，boxmesh 位置是幾片的平均（11-25 試過，偏 1 cm）。
        sp_ = spec[pn]
        lv = np.asarray(sp_["vertices"], float)
        tr = np.asarray(sp_["translation"], float)[:2]
        cf = tr[0]
        side = np.sign(lv[:, 0].mean()) if abs(lv[:, 0].mean()) > 1e-6 else 1.0
        hem = tr[1] + lv[:, 1].min()
        on_cf = np.abs(lv[:, 0]) < 1e-3
        top_cf = tr[1] + (lv[on_cf, 1].max() if on_cf.any() else lv[:, 1].max())
        # 這片在 UV 圖上的區域：面重心落在哪個 island
        cen = canvas.px(np.array([vt[ft[k]].mean(0) for k in fs])).astype(int)
        cen = np.clip(cen, 0, canvas.size - 1)
        ids = canvas.islands[cen[:, 1], cen[:, 0]]
        isl = np.bincount(ids[ids > 0]).argmax()
        mask = canvas.islands == isl
        # 像素 → UV → 版片座標
        yy, xx = np.nonzero(mask)
        uv = np.c_[(xx + 0.5) / canvas.size, 1 - (yy + 0.5) / canvas.size]
        L = M[:2].T  # uv = L @ [x,y] + t
        t = M[2]
        xy = np.linalg.solve(L, (uv - t).T).T
        out[pn] = {"M": M, "cf": cf, "side": side, "hem": hem, "top": top_cf, "mask": (yy, xx),
                   "u": (xy[:, 0] - cf) * side, "w": xy[:, 1] - hem, "faces": fs}
    return out


def neck_w(frames, pn, box, seg):
    """前中心最上面那一點（領口最低處）離下襬多高（版片規格讀的）。"""
    return float(frames[pn]["top"] - frames[pn]["hem"])


def buttons_at(cfg, top_w):
    ys, w = [], top_w - cfg["button_top"]
    while w > cfg["button_bottom"] - 1e-6:
        ys.append(w)
        w -= cfg["button_pitch"]
    return ys


def paint(code, cfg, ctx, box, with_buttons):
    """在 h、shade、stitch（原地）和顏色圖上加細節；回傳 (顏色覆蓋 dict（像素 → 顏色）, 說明)。"""
    canvas, h, shade, stitch = ctx["canvas"], ctx["h"], ctx["shade"], ctx["stitch"]
    spec = json.loads((ctx["folder"] / f"{ctx['name']}_specification.json").read_text(encoding="utf-8"))["pattern"]["panels"]
    frames = panel_frames(ctx["v"], ctx["vt"], ctx["fv"], ctx["ft"], ctx["seg"], ctx["stitches"], canvas, box, spec)
    color_over = []
    info = {}
    for pn, fr in frames.items():
        yy, xx = fr["mask"]
        u, w = fr["u"], fr["w"]
        if code == "z02":
            half = cfg["placket_half"]
            d = np.abs(u - half)  # 門襟邊（摺邊）：陷一條、亮一條
            h[yy, xx] += -0.7 * np.exp(-(d / 0.08) ** 2) + 0.3 * (u < half)
            shade[yy, xx] -= 0.18 * np.exp(-(d / 0.06) ** 2)
            a = w  # 車縫沿前中心方向
            m, dh = st.stitch_line(np.abs(u - (half - cfg["placket_stitch"])), a, 0.0)
            stitch[yy, xx] |= m
            h[yy, xx] += dh
            if with_buttons:
                top = neck_w(frames, pn, box, ctx["seg"])
                ys = buttons_at(cfg, top)
                info["扣子"] = [round(y, 1) for y in ys]
                for by in ys:
                    r = np.hypot(u, w - by)
                    inside = r < cfg["button_r"]
                    dome = np.clip(1 - (r / cfg["button_r"]) ** 2, 0, 1)
                    holes = np.zeros_like(r, bool)
                    for hx, hy in ((0.14, 0.14), (-0.14, 0.14), (0.14, -0.14), (-0.14, -0.14)):
                        holes |= np.hypot(u - hx, w - by - hy) < 0.06
                    h[yy[inside], xx[inside]] += 1.6 + 0.8 * dome[inside] - 1.2 * holes[inside]
                    rim = np.abs(r - cfg["button_r"] * 0.78) < 0.05
                    h[yy[rim & inside], xx[rim & inside]] += 0.3
                    shade[yy, xx] -= 0.35 * np.exp(-((r - cfg["button_r"] - 0.08) / 0.07) ** 2)  # 扣子邊的影子
                    color_over.append((yy[inside], xx[inside], np.where(holes[inside], 0.6, 1.0)))
        if code == "z01":
            pk = dict(cfg["pocket"])
            if any(s_.startswith("wb_") for s_ in ctx["seg"]):  # 有下襬羅紋：口袋底邊坐在羅紋接縫上，整個口袋往下移
                shift = pk["bottom"] - 0.6
                pk["bottom"] -= shift
                pk["opening_to"] -= shift
            b, ht, th, bh, op = pk["bottom"], pk["height"], pk["top_half"], pk["bottom_half"], pk["opening_to"]
            top = b + ht
            # 斜開口：從 (th, top) 到 (bh, op)；口袋範圍＝底邊以上、頂邊以下、斜線內側、外側直邊以內
            def slant(uu):
                return top + (uu - th) * (op - top) / (bh - th)
            inside = (w > b) & (w < top) & (u < bh) & ((u < th) | (w < slant(u)))
            h[yy[inside], xx[inside]] += 0.3
            # 縫起來的邊：頂邊（u<th）、底邊、外側直邊（w<op）→ 接縫＋雙車縫
            d_top = np.where(u < th, np.abs(w - top), np.inf)
            d_bot = np.where(u < bh, np.abs(w - b), np.inf)
            d_side = np.where(w < op, np.abs(u - bh), np.inf)
            d = np.minimum(np.minimum(d_top, d_bot), d_side)
            h[yy, xx] += -0.8 * np.exp(-(d / 0.08) ** 2)
            shade[yy, xx] -= 0.15 * np.exp(-(d / 0.06) ** 2)
            for at in (0.15, 0.75):
                m = (np.abs(d - at) < 0.05) & ((np.where(d == d_top, u, w) % 0.42) < 0.3) & inside
                stitch[yy[m], xx[m]] = True
            # 開口（斜線）：摺邊微微隆起，外側一條陰影（手插進去的縫）
            nrm = np.hypot(1, (op - top) / (bh - th))
            ds = (w - slant(u)) / nrm  # >0：在斜線外側
            band = (u > th - 0.5) & (u < bh + 0.5) & (w > op - 0.5) & (w < top + 0.5)
            h[yy[band], xx[band]] += 0.5 * np.exp(-((ds[band] + 0.3) / 0.25) ** 2)
            shade[yy[band], xx[band]] -= 0.45 * np.exp(-((ds[band] - 0.15) / 0.18) ** 2)
            m = band & (np.abs(ds + 0.8) < 0.05) & ((w % 0.42) < 0.3)
            stitch[yy[m], xx[m]] = True
            info["口袋"] = f"寬 {2 * bh:.0f}、高 {ht:.0f}、底邊離下襬 {b:.0f} cm"
    return color_over, info, frames


def attach(point_uw, fr, pn, box, x, fv, seg):
    """版片座標 (u, w) → 縫好網格上的 (面編號, 重心座標, 3D 位置, 法向)。"""
    px = fr["cf"] + point_uw[0] * fr["side"]
    py = fr["hem"] + point_uw[1]
    p = np.array([px, py])
    best = None
    for k in fr["faces"]:
        a, b_, c = box[fv[k], :2]
        T = np.array([b_ - a, c - a]).T
        if abs(np.linalg.det(T)) < 1e-9:
            continue
        l1, l2 = np.linalg.solve(T, p - a)
        bc = np.array([1 - l1 - l2, l1, l2])
        err = -min(bc.min(), 0)
        if best is None or err < best[0]:
            best = (err, k, bc)
    _, k, bc = best
    bc = np.clip(bc, 0, None); bc /= bc.sum()
    tri = x[fv[k]]
    pos = bc @ tri
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n)
    if n[2] < 0:  # 前片朝前（+z）
        n = -n
    return int(k), bc.tolist(), pos, n


def cylinder(center, normal, r, hgt, seg=16, z0=0.05):
    """沿 normal 的圓柱（扣子），回傳 (頂點, 面)。"""
    n = normal / np.linalg.norm(normal)
    a = np.cross(n, [0, 1, 0]); a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])
    b = np.cross(n, a)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
    bot = center + n * z0 + r * ring
    top = center + n * (z0 + hgt) + 0.92 * r * ring
    V = np.vstack([bot, top, center + n * z0, center + n * (z0 + hgt * 1.15)])
    F = []
    for i in range(seg):
        j = (i + 1) % seg
        F += [[i, j, seg + j], [i, seg + j, seg + i], [2 * seg + 1, seg + i, seg + j], [2 * seg, j, i]]
    return V, np.array(F)


def tube(path, r, seg=8):
    V, F = [], []
    for i, p in enumerate(path):
        t = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        a = np.cross(t, [0, 0, 1.0]); a /= np.linalg.norm(a)
        b = np.cross(t, a)
        for k in range(seg):
            ang = 2 * np.pi * k / seg
            V.append(p + r * (np.cos(ang) * a + np.sin(ang) * b))
    for i in range(len(path) - 1):
        for k in range(seg):
            q = (k + 1) % seg
            F += [[i * seg + k, (i + 1) * seg + k, (i + 1) * seg + q], [i * seg + k, (i + 1) * seg + q, i * seg + q]]
    return np.array(V), np.array(F)


def main(code, mesh, folder, out, design=None):
    folder, out = Path(folder), Path(out)
    cfg = DETAILS[code]
    name = folder.name
    import igl
    box, _ = igl.read_triangle_mesh(str(folder / f"{name}_boxmesh.obj"))
    keep = {}

    def extra_for(with_buttons):
        def extra(ctx):
            over, info, frames = paint(code, cfg, ctx, box, with_buttons)
            keep["over"], keep["frames"], keep["ctx"] = over, frames, ctx
            return info
        return extra

    from PIL import Image
    for prefix, with_buttons in (("seam", False), ("detail", True)):
        st.main(mesh, folder, out, 2048, design, extra=extra_for(with_buttons), prefix=prefix)
        # 灰階 → 真的顏色：布色乘上灰階明暗；扣子、帽繩用自己的顏色
        img = np.asarray(Image.open(out / f"{prefix}_color.png")).astype(float) / 255
        g = img[..., :1] / st.GRAY[0]
        col = np.clip(g * np.array(cfg["fabric"])[None, None], 0, 1)
        if with_buttons:
            for yy, xx, lum in keep["over"]:
                col[yy, xx] = np.array(cfg["button"]) * lum[:, None] * np.clip(g[yy, xx], 0.7, 1.2)
        Image.fromarray((col * 255).astype(np.uint8)).save(out / f"{prefix}_color.png")

    # B：立體零件
    ctx, frames = keep["ctx"], keep["frames"]
    x, fv, seg = ctx["v"], np.asarray(ctx["fv"]), ctx["seg"]
    parts, Vs, Fs, groups = [], [], [], []
    nv = 0

    def add(V, F, label):
        nonlocal nv
        Vs.append(V); Fs.append(F + nv); groups.append((label, nv, len(V))); nv += len(V)

    if code == "z02":
        pn = "right_ftorso" if "right_ftorso" in frames else next(iter(frames))
        top = neck_w(frames, pn, box, seg)
        for i, by in enumerate(buttons_at(cfg, top)):
            k, bc, pos, n = attach((0.0, by), frames[pn], pn, box, x, fv, seg)
            V, F = cylinder(pos, n, cfg["button_r"], 0.28)
            add(V, F, f"button_{i}")
            parts.append({"name": f"button_{i}", "kind": "button", "face": k, "bary": bc, "offset_cm": 0.05,
                          "radius_cm": cfg["button_r"], "uw_cm": [0.0, round(by, 1)], "color": cfg["button"]})
    if code == "z01":
        # 帽繩：帽子開口那圈布邊，前中心左右各 cord_from_cf，最低那段
        loops = igl.all_boundary_loop(fv)
        hood = np.array(["hood" in s for s in seg])
        lp = max(loops, key=lambda L: np.mean([hood[i] for i in L]))
        lp = np.array(lp)
        for s in (1, -1):
            cand = lp[(np.sign(x[lp, 0]) == s)]
            cand = cand[x[cand, 1] < x[lp, 1].min() + 8]
            start = cand[np.abs(np.abs(x[cand, 0]) - cfg["cord_from_cf"]).argmin()]
            p0 = x[start].copy()
            path = [p0]
            for dy in np.arange(1, cfg["cord_len"] + 1):
                q = np.array([p0[0], p0[1] - dy, p0[2] + 3])
                d2, fi, cp = igl.point_mesh_squared_distance(q[None], x, fv)
                fi, cp = np.atleast_1d(fi), np.atleast_2d(cp)
                tri = x[fv[fi[0]]]
                nn = np.cross(tri[1] - tri[0], tri[2] - tri[0]); nn /= np.linalg.norm(nn)
                if nn[2] < 0:
                    nn = -nn
                path.append(cp[0] + nn * (cfg["cord_r"] + 0.25))
            path = np.array(path)
            V, F = tube(path[:-3], cfg["cord_r"])
            add(V, F, f"cord_{'L' if s > 0 else 'R'}")
            V, F = tube(path[-4:], cfg["cord_r"] * 1.35)
            add(V, F, f"aglet_{'L' if s > 0 else 'R'}")
            parts.append({"name": f"cord_{'L' if s > 0 else 'R'}", "kind": "cord", "anchor_vertex": int(start),
                          "length_cm": cfg["cord_len"], "radius_cm": cfg["cord_r"], "color": cfg["cord"],
                          "note": "只固定上端（帽子開口的頂點），其餘要 engine 當繩子模擬；這裡是靜止 T-pose 時貼著衣服垂下的樣子"})
    if Vs:
        V, F = np.vstack(Vs) / 100.0, np.vstack(Fs)
        lines = []
        for (label, s0, n), Fg in zip(groups, Fs):  # 每個物件的面緊接在它的頂點後面（OBJLoader 才會分成不同物件）
            lines.append(f"o {label}\n")
            lines += ["v %.6f %.6f %.6f\n" % tuple(p) for p in V[s0:s0 + n]]
            lines += ["f %d %d %d\n" % tuple(t + 1) for t in Fg]
        (out / "parts.obj").write_text("".join(lines))
        (out / "parts_attach.json").write_text(json.dumps({
            "units": "m（offset、radius、length 是 cm）", "garment": "同資料夾 garment_uv.obj（頂點順序同 engine 的衣服 OBJ）",
            "parts": parts, "n_vertices": int(len(V)),
            "說明": "button：位置＝face 三個頂點 × bary，沿面法向往外 offset；每幀照衣服目前的頂點重算位置與方向即可跟著布動",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"零件 {len(parts)} 個、{len(V)} 頂點 → {out / 'parts.obj'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
