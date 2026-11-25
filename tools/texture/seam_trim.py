# -*- coding: utf-8 -*-
r"""在衣服的 UV 貼圖上畫「全部的縫線」：接縫凹痕、車縫線、袖口羅紋、下襬／袖口反摺車縫，另外輸出法線貼圖（假凹凸）。
網格、頂點數、量測都不動，只多兩張貼圖。領口的羅紋帶沿用 neck_trim.py（圓領才加，帽 T、翻領不加）。

哪些是縫線：一條邊兩側的三角形屬於不同版片（sim_segmentation.txt）。但 GarmentCode 把衣身切成左右兩半、
袖子切成前後兩片，真的衣服這幾條不是縫線，不畫：
  左右衣身的中線（left_*torso ↔ right_*torso）、袖子上緣（*_sleeve_f ↔ *_sleeve_b 在手臂上方那條）
畫什麼（距離都換成 cm，照真的 T 恤、帽 T）：
  每條縫線     兩側往下陷一條細縫＋縫份撐起的微微鼓起，顏色深一點
  袖籠、肩線、帽子／領子接縫、袖口接縫   衣身那側（或袖子那側）0.15 cm 一條車縫
  袖口片（CuffBand）  整片縱向羅紋（間距 0.3 cm）
  下襬、沒有袖口的袖口   反摺：離邊 2.0、2.6 cm 兩條車縫（雙針），邊緣圓起
跑法（GarmentCode 的 .venv）：
    python tools\texture\seam_trim.py <縫好的網格.obj（cm）> <GarmentCode 縫合資料夾> <輸出資料夾> [貼圖邊長，預設 2048] [design.yaml]
  縫合資料夾要有 <名稱>_boxmesh.obj（UV、面）、_sim_segmentation.txt、_vertex_labels.yaml；網格頂點順序要和 boxmesh 一樣。
輸出：garment_uv.obj（公尺、帶 UV，座標同輸入）、seam_color.png、seam_normal.png、plain_color.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neck_trim import (BAND_CM, GRAY, RIB_PITCH_CM, STYLES, Canvas, neckline_edges, order_loop,  # noqa: E402
                       read_obj, stitch_line, trim_style)

TOPSTITCH_CM = 0.15     # 車縫離接縫的距離
HEM_STITCH_CM = (2.0, 2.6)  # 下襬雙針車縫離邊的距離
SEAM_W = 0.12  # 接縫凹痕的寬度（cm，高斯的 σ）
NORMAL_STRENGTH = 0.16  # 法線貼圖的強度（neck_trim 是 0.08）
THREAD_COLOR = [0.36, 0.37, 0.38]  # 車縫線顏色（灰階，乘上布的顏色）
STITCHED = ("armhole", "shoulder", "neck", "cuff", "hem_band")  # 要畫車縫的縫線種類


def panel_of(face, seg):
    names = [seg[i] for i in face if not seg[i].startswith("stitch")]
    return max(set(names), key=names.count) if names else ""


def face_panels(fv, seg, stitches):
    """每個面屬於哪片版片。三個頂點都在縫線上的面（縫線交會處、轉角）用頂點身上的縫線推：
    每條縫線只連兩片版片（specification.json 的 stitches），一個頂點可能屬於的版片＝它身上每條縫線兩端版片的聯集，
    面的版片＝三個頂點可能版片的交集。例：肩點（袖子上緣、前後袖籠）∩ 後袖籠上的點 ∩ 肩線上的點 → 只剩後身。
    交集不是剛好一片的，再跟「隔著同一片可能版片」的鄰面。
    11-25 林宇鑫發現：沒處理時肩點附近的小三角形被當成別片，肩點畫成一個框；只看鄰面時又判錯成前身，縫線歪一段。"""
    pair = {f"stitch_{i}": {st[0]["panel"], st[1]["panel"]} for i, st in enumerate(stitches)
            if len(st) >= 2 and isinstance(st[0], dict) and isinstance(st[1], dict)}

    def can(i):
        if not seg[i].startswith("stitch"):
            return {seg[i]}
        out = set()
        for t in seg[i].split(","):
            out |= pair.get(t, set())
        return out

    lab, todo = [], []
    for k, f in enumerate(fv):
        name = panel_of(f, seg)
        if not name:
            cand = can(f[0]) & can(f[1]) & can(f[2])
            name = next(iter(cand)) if len(cand) == 1 else ""
            if not name:
                todo.append((k, cand))
        lab.append(name)
    if todo:
        by_edge = {}
        for k, f in enumerate(fv):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                by_edge.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append(k)
        for _ in range(5):
            for k, cand in todo:
                if lab[k]:
                    continue
                f = fv[k]
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    other = [lab[q] for q in by_edge[(min(f[a], f[b]), max(f[a], f[b]))] if q != k and lab[q]]
                    if other and (not cand or other[0] in cand):
                        lab[k] = other[0]
                        break
    return lab


def kind(a, b, mid_y, arm_y, armpit_y):
    """縫線種類；None＝真的衣服沒有這條縫。a、b：兩側版片名稱。"""
    pa, pb = sorted((a, b))
    strip = lambda p: p.replace("left_", "").replace("right_", "").replace("sl_", "")  # noqa: E731
    sa, sb = strip(pa), strip(pb)
    if sa == sb and "hood" not in sa:          # 左右兩半的中線（帽子的中線是真的）
        return None
    if {sa, sb} == {"sleeve_f", "sleeve_b"}:
        return None if mid_y > arm_y else "underarm"  # 手臂上方那條不是縫線，腋下那條是
    if "cuff" in sa or "cuff" in sb:
        return "cuff" if ("sleeve" in sa or "sleeve" in sb) else None
    if sa.startswith("wb_") or sb.startswith("wb_"):  # 下襬羅紋（腰帶）：和衣身的接縫；羅紋前後片之間是側縫
        return "hem_band" if ("torso" in sa or "torso" in sb) else "side"
    if "sleeve" in sa or "sleeve" in sb:
        return "armhole"
    if {sa, sb} == {"ftorso", "btorso"}:
        return "side" if mid_y < armpit_y + 1 else "shoulder"
    if "hood" in sa or "hood" in sb or "collar" in sa or "collar" in sb:
        return "neck"
    return "other"


def seams(v, fv, ft, seg, stitches):
    """每條縫線：[(種類, 頂點 i, j, 這側版片, 這側 uv i, uv j)]，一條縫線兩側各一筆。"""
    sleeve = np.array(["sleeve" in s for s in seg])
    torso = np.array(["torso" in s for s in seg])
    near_body = sleeve & (np.abs(v[:, 0]) < np.abs(v[torso, 0]).max() + 4)
    armpit_y = v[near_body, 1].min()  # 腋下：袖子靠身體那端最低的點（同 spec_batch.measure_3d）
    by_edge = {}
    for f, t, name in zip(fv, ft, face_panels(fv, seg, stitches)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            by_edge.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append((name, {f[a]: t[a], f[b]: t[b]}))
    out = []
    for (i, j), sides in by_edge.items():
        if len(sides) != 2 or sides[0][0] == sides[1][0]:
            continue
        mid = (v[i] + v[j]) / 2
        near = sleeve & (np.abs(v[:, 0] - mid[0]) < 3)  # 這個位置的手臂中心高度
        arm_y = v[near, 1].mean() if near.any() else mid[1]
        k = kind(sides[0][0], sides[1][0], mid[1], arm_y, armpit_y)
        if k is None:
            continue
        for name, uv in sides:
            out.append((k, i, j, name, uv[i], uv[j]))
    return out


def arcs(v, edges):
    """一串邊（可能分好幾段）依相接排好，回傳 [(uv a, uv b, 弧長 a, 弧長 b)]（弧長 cm，給車縫虛線用）。"""
    tup = [(e[1], e[2], e[4], e[5]) for e in edges]
    segs, s0, last = [], 0.0, None
    for i, j, ti, tj in order_loop(tup):
        if last is not None and i != last:
            s0 += 1.0  # 不相接的段落之間隔開
        seg = float(np.linalg.norm(v[i] - v[j]))
        segs.append((ti, tj, s0, s0 + seg))
        s0 += seg
        last = j
    return segs


def boundary_side(v, fv, ft, seg, pick, stitches):
    """網格邊界（只有一個面的邊）中，pick(版片, 中點) 為真的：[(uv a, uv b, 弧長 a, 弧長 b)]"""
    cnt = {}
    for f, t, name in zip(fv, ft, face_panels(fv, seg, stitches)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cnt.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append((f[a], f[b], t[a], t[b], name))
    es = [e[0] for e in cnt.values() if len(e) == 1 and pick(e[0][4], (v[e[0][0]] + v[e[0][1]]) / 2, e[0])]
    return arcs(v, [(None, i, j, None, ti, tj) for i, j, ti, tj, _ in es])


def igl_loops(fv):
    """網格的邊界圈（頂點編號串）。"""
    import igl
    return igl.all_boundary_loop(np.asarray(fv, np.int64))


def main(mesh, folder, out, size=2048, design=None, extra=None, prefix="seam"):
    """extra：之後再加細節的函式（detail_trim.py 用），收到 dict（canvas、h、shade、stitch、v、vt、fv、ft、seg、folder、name），
    直接改 h、shade、stitch（原地），回傳要寫進說明的 dict。prefix：貼圖檔名開頭。"""
    size, folder, out = int(size), Path(folder), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = folder.name
    box = folder / f"{name}_boxmesh.obj"
    _, vt, fv, ft = read_obj(box)
    v = np.array([[float(t) for t in ln.split()[1:4]] for ln in open(mesh) if ln.startswith("v ")])
    assert len(v) == len(read_obj(box)[0]), "網格和 boxmesh 頂點數不同"
    seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
    canvas = Canvas(v, vt, fv, ft, size)
    h = np.zeros((size, size))
    shade = np.ones((size, size))
    stitch = np.zeros((size, size), bool)

    # 1. 縫線：兩側凹痕＋縫份鼓起
    stitches = json.loads((folder / f"{name}_specification.json").read_text(encoding="utf-8"))["pattern"]["stitches"]
    sm = seams(v, fv, ft, seg, stitches)
    d_all, _, _ = canvas.field(*canvas.draw_lines(arcs(v, sm)))
    # 凹痕寬約 2～3 mm、縫份撐起約 1 cm 寬；比領口那套（neck_trim）深，因為縫線要從試穿的距離看得出來
    h += -1.3 * np.exp(-(d_all / SEAM_W) ** 2) + 0.5 * np.exp(-((d_all - 0.55) / 0.4) ** 2)
    shade -= 0.38 * np.exp(-(d_all / SEAM_W) ** 2)
    shade += 0.05 * np.exp(-((d_all - 0.55) / 0.4) ** 2)
    # 2. 車縫：衣身那側（袖籠、肩線、領／帽接縫），袖口接縫畫在袖子那側
    def stitched_side(e):
        k, p = e[0], e[3]
        if k == "cuff":
            return "sleeve" in p
        if k == "shoulder":
            return "btorso" in p
        return k in STITCHED and "torso" in p
    top = [e for e in sm if stitched_side(e)]
    if top:
        d, a, _ = canvas.field(*canvas.draw_lines(arcs(v, top)))
        m, dh = stitch_line(d, a, TOPSTITCH_CM)
        stitch |= m
        h += dh
    # 3. 袖口羅紋：整片 cuff 版片，羅紋沿手臂方向一條一條（沿袖口接縫的弧長變化）
    cuff_seam = [e for e in sm if e[0] == "cuff" and "cuff" in e[3]]
    has_cuff = bool(cuff_seam)
    if cuff_seam:
        _, a, area = canvas.field(*canvas.draw_lines(arcs(v, cuff_seam)))
        rib = 0.5 + 0.5 * np.cos(2 * np.pi * a / RIB_PITCH_CM)
        h[area] += 0.35 + 0.25 * rib[area]
        shade[area] *= 0.93 + 0.05 * rib[area]
    # 3b. 下襬羅紋：整片 wb_ 版片畫羅紋（沿接縫的弧長變化＝一條一條直的）
    band_seam = [e for e in sm if e[0] == "hem_band" and e[3].startswith("wb_")]
    if band_seam:
        _, a, area = canvas.field(*canvas.draw_lines(arcs(v, band_seam)))
        rib = 0.5 + 0.5 * np.cos(2 * np.pi * a / RIB_PITCH_CM)
        h[area] += 0.35 + 0.25 * rib[area]
        shade[area] *= 0.93 + 0.05 * rib[area]
    # 4. 反摺：下襬（衣身最低的布邊），沒有袖口的話袖口也是
    # 下襬＝最低的那一圈布邊（同 spec_measure.length）裡，在版片（UV）上接近水平的邊
    # （開襟衫那圈連著前開口，前開口是直的，自動排除；穿在身上前片下襬中間會被肚子頂高，不能用高度判斷）
    loops = igl_loops(fv)
    hem_v = set(int(i) for i in min(loops, key=lambda lp: v[lp, 1].mean()))
    flat = lambda e: abs(vt[e[2], 1] - vt[e[3], 1]) < 0.5 * abs(vt[e[2], 0] - vt[e[3], 0])  # noqa: E731
    hem = boundary_side(v, fv, ft, seg, lambda p, mid, e: "torso" in p and e[0] in hem_v and e[1] in hem_v and flat(e), stitches)
    if not has_cuff:  # 袖口：袖子那兩圈布邊（離身體最遠）
        cuffs = sorted(loops, key=lambda lp: -abs(v[lp, 0].mean()))[:2]
        cv = {i for lp in cuffs for i in lp}
        hem += boundary_side(v, fv, ft, seg, lambda p, mid, e: "sleeve" in p and e[0] in cv and e[1] in cv, stitches)
    if hem:
        d, a, _ = canvas.field(*canvas.draw_lines(hem))
        h -= 0.6 * np.exp(-((d - 0.06) / 0.12) ** 2)  # 反摺的圓邊
        shade -= 0.15 * np.exp(-((d - 0.06) / 0.1) ** 2)
        for at in HEM_STITCH_CM:
            m, dh = stitch_line(d, a, at)
            stitch |= m
            h += dh
    # 5. 圓領羅紋帶（照 neck_trim 的規則：帽 T、翻領不加）
    design = yaml.safe_load(open(design, encoding="utf-8")) if design else None
    style = trim_style(design.get("design", design))[0] if design else None
    if style and not style.startswith("turtle"):
        lab = yaml.safe_load(open(folder / f"{name}_vertex_labels.yaml", encoding="utf-8"))
        collar = {i for k, ids in lab.items() if "collar" in k for i in ids}
        segs, s0 = [], 0.0
        for i, j, ti, tj in order_loop(neckline_edges(fv, ft, collar)):
            L = float(np.linalg.norm(v[i] - v[j]))
            segs.append((ti, tj, s0, s0 + L))
            s0 += L
        d, a, _ = canvas.field(*canvas.draw_lines(segs))
        band = d < BAND_CM
        rib = 0.5 + 0.5 * np.cos(2 * np.pi * a / RIB_PITCH_CM)
        h[band] += 0.35 + 0.25 * rib[band]
        h -= 0.5 * np.exp(-((d - 0.08) / 0.10) ** 2) + 0.9 * np.exp(-((d - BAND_CM) / 0.07) ** 2)
        shade[band] *= 0.93 + 0.05 * rib[band]
        shade -= 0.25 * np.exp(-((d - BAND_CM) / 0.05) ** 2)
        for at in STYLES["single"]:
            m, dh = stitch_line(d, a, at)
            stitch |= m
            h += dh

    extra_info = {}
    if extra is not None:
        extra_info = extra(dict(canvas=canvas, h=h, shade=shade, stitch=stitch, v=v, vt=vt, fv=fv, ft=ft, seg=seg,
                                folder=folder, name=name, stitches=stitches)) or {}
    # 存檔（同 neck_trim.save，檔名不同）
    color = np.clip(GRAY[None, None] * shade[..., None], 0, 1)
    color[stitch] = THREAD_COLOR
    color[~canvas.mask] = GRAY
    h = h + 0.25 * stitch
    gy, gx = np.gradient(ndimage.gaussian_filter(h, 0.6), canvas.cm_per_px)
    nrm = np.dstack([-gx * NORMAL_STRENGTH, gy * NORMAL_STRENGTH, np.ones_like(h)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Image.fromarray((color * 255).astype(np.uint8)).save(out / f"{prefix}_color.png")
    Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save(out / f"{prefix}_normal.png")
    Image.fromarray(np.full((size, size, 3), (GRAY * 255).astype(np.uint8))).save(out / "plain_color.png")
    # 帶 UV 的網格（公尺；v 換成縫好的位置，vt、f 照抄 boxmesh）
    k, lines = 0, []
    for ln in open(box):
        if ln.startswith("v "):
            lines.append("v %.6f %.6f %.6f\n" % tuple(v[k] / 100))
            k += 1
        elif ln.startswith(("vt ", "f ")):
            lines.append(ln)
    (out / "garment_uv.obj").write_text("".join(lines))
    from collections import Counter
    n = Counter(e[0] for e in sm)
    info = {"縫線（邊數，兩側各算一次）": dict(n), "車縫的縫線邊": len(top), "袖口羅紋": has_cuff, "反摺邊": len(hem),
            "圓領羅紋帶": bool(style and not style.startswith("turtle")), "UV 每像素 mm": round(canvas.cm_per_px * 10, 2), **extra_info}
    (out / f"{prefix}_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
