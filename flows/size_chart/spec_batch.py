# -*- coding: utf-8 -*-
r"""照尺寸表打版的整批版：網路衣服的每一件、每一個尺碼都打版、縫到 SMPL-X 中性身體上，
再量縫好的 3D 衣服，和尺寸表（廠商平放量的 cm）比差多少。

款式：每件借一件款式相近的 AI 結果（TEMPLATES），套上領深校正、合身款改一般款；
     胸寬、衣長、袖長、肩寬照尺寸表（fit2，2D 版片上量到差 0.2 cm 內，肩寬 0.5 cm 內）。
     2D 版片＝尺寸表，縫一次就好（2026-11-24 拿掉「量 3D、差太多就修 2D 目標重縫」的迴圈：
     那個迴圈是在補量法的偏差，量法修好後 3D ≈ 版片，再修只會把版片越修越偏）。
袖山抽皺（sleeve.connect_ruffle）：AI 給 1.11～1.18（像泡泡袖），改成落肩 1.0、一般肩 1.05。
縫合：縫好、攤平尖角之後，從攤平的位置再縫兩次（sew_finish，彎曲剛度 3），不然落肩的肩角會被攤平縮掉 3～5 cm。
3D 量法（縫好、T-pose、沒有重力，單位 cm）。尺寸表是平放「沿著布」量的，所以 3D 也沿著布面量，
距離用 spec_measure.surface_dist（精確最短距離；之前的 heat_geodesic 會偏短約 2 cm）：
  胸寬  腋下往下 2.5 cm 那一圈衣身的周長 ÷ 2（等於平放時的寬度）
  衣長  肩頸點 → 後片下襬最近的一點，只走後片（等於後片攤平後垂直量），左右平均（2026-11-24 改）
  袖長  肩點 → 袖口邊最近的一點，只走袖子和袖口（＝沿袖片上緣量），左右平均（2026-11-24 改，原本起點是衣身最外側的點）
  肩寬  左肩點 → 右肩點，只走後片（2026-11-24 改：肩點改用縫線資料找真的肩點，見 spec_measure.py）
  下襬圍 下襬那一圈的長度（尺寸表有才比）
  另外記「袖口伸到多遠」＝肩寬÷2＋袖長（日本尺寸表的「裄丈」），看整隻袖子跟真的衣服差多少
  診斷：「肩角被壓」＝肩點各內縮 3 cm 量再加回版片那兩段 − 正式肩寬（縫合模擬把肩點附近壓縮的量，之後要想辦法修）

跑法（在 repo 根目錄；這台筆電要從英文路徑副本跑，libigl 開不了中文路徑）：
    .venv\Scripts\python flows\size_chart\spec_batch.py [衣服代號…，預設全部]     一件一件做（可以開好幾個程序分開跑）
    .venv\Scripts\python flows\size_chart\spec_batch.py --remeasure                  量法改了：用縫好的網格重量，不重縫
    .venv\Scripts\python flows\size_chart\spec_batch.py --summary                    把做好的合成 總表.json（網頁用）
輸出：output/spec_to_pattern/_batch/<代號>/<尺碼>/（縫好的 .obj、量測.json）、_batch/總表.json
     照尺寸表第 1 次就畫不出版片的尺碼（太小）：只寫 跳過.json（原因、尺寸表），不中斷，繼續下一個
"""
import base64
import copy
import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import igl
import numpy as np
import yaml
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
BODY_FIT = HERE.parent / "body_fit"
sys.path.insert(0, str(BODY_FIT))
sys.path.insert(0, str(HERE))
import spec_to_pattern as sp  # noqa: E402  （會切到 GarmentCodeRC 根目錄）
import spec_measure as smeas  # noqa: E402
import drop_shoulder  # noqa: E402,F401  註冊落肩版型 DropShoulderShirt
from assets.bodies.body_params import BodyParameters  # noqa: E402
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from export_for_physics import prepare_smplx_body  # noqa: E402
from export_neutral import build, load_design  # noqa: E402
from neckline import fix_neck_depth  # noqa: E402
from export_neutral import RESOLUTION, SEW  # noqa: E402
from sew import mesh_edges, orient_faces, rest_lengths, sew, unfold  # noqa: E402

DATA = HERE / "網路衣服_尺寸資料"
AI = sp.ROOT / "results" / "chatgarment_ai"
OUT = sp.ROOT / "output" / "spec_to_pattern" / "_batch"
# 每件網路衣服的款式（領型、袖型、帽子、前襟…）：2026-11-15 起用這 7 件自己的推論結果（bf16_z，3090 bf16）
TEMPLATES = {
    "z01_帽T": "bf16_z/z01_black_hoodie",
    "z02_襯衫": "bf16_z/z02_blue_cropped_shirt",
    "z03_針織衫": "bf16_z/z03_beige_oversized_sweater",
    "z04_POLO": "bf16_z/z04_charcoal_knit_polo",
    "z05_開襟衫": "bf16_z/z05_olive_knit_cardigan",
    "z06_長袖T": "bf16_z/z06_white_longsleeve_tee",
    "z07_U領長袖T": "bf16_z/z07_navy_squareneck_longsleeve_tee",
}
RUFFLE_DROP = 1.0  # 落肩款袖山不吃縫（衣身肩線直接延伸到手臂上，平接）
RUFFLE_REGULAR = 1.05  # 一般肩袖山比袖籠長 5%（真的 T 恤約 0～7%）
RESEW_BEND = 3.0  # 攤平尖角後再縫一次用的彎曲剛度（第一次縫是 SEW["bend"]＝1）
RESEW_ITERS = 200
RESEW_ROUNDS = 2  # 再縫幾次（z06 M：1 次 −1.4、2 次 −0.5、3 次 −0.9；z04 M：−4.1、−2.8、−2.5）
RESEW_MAX_ROUNDS = 4  # 2026-11-25：2 次後肩角還被壓超過 SQUEEZE_OK，就繼續再縫，最多到這麼多次；留壓縮最小的那次
SQUEEZE_OK = 0.5  # cm，「肩角被壓(診斷)」在這以內就不再縫
MAX_VERTS = 3900  # engine 的頂點上限是 4,000（2026-11-09 和紀泓宇確認），留 100 的餘裕


def pick_resolution(design, name, body_yaml):
    """頂點間距：先用 RESOLUTION（2 cm），頂點數超過 MAX_VERTS 就放大間距再打一次網格（只打 box mesh、不縫，幾秒）。
    頂點數大約和間距的平方成反比。帽 T、大號襯衫用 2 cm 會到 4,200～4,900。回傳 (間距, 頂點數)。"""
    res = RESOLUTION
    for _ in range(5):
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, paths = sp.dm.generate(design, body_yaml, name, resolution=res)
        n = len(igl.read_triangle_mesh(str(paths.g_box_mesh))[0])
        if n <= MAX_VERTS:
            return res, n
        res *= float(np.sqrt(n / (MAX_VERTS * 0.98)))
    return res, n


def sew_finish(x, f, box, folder, name, body_mesh, squeeze=None):
    """縫好之後：攤平尖角，再「從攤平的位置再縫一次＋攤平」RESEW_ROUNDS 次。回傳 (新位置, 每次攤平的說明, 每次的肩角壓縮)。
    squeeze：給「位置 → 肩角被壓幾 cm」的函式（spec_measure.shoulder_inset − 正式肩寬）時，2 次後還超過 SQUEEZE_OK
      就繼續再縫，最多 RESEW_MAX_ROUNDS 次，回傳壓縮最小的那次（再縫不一定越縫越好：z06 M 第 3 次反而變差）。
      這個判斷只看布被壓了多少，不看尺寸表，所以不是拿目標數字去湊。
    為什麼要再縫：落肩款的肩點（四條縫線交會處）縫好時會翻折，unfold 把那幾個點往鄰居平均位置拉，
    肩角附近的布被縮到原長的 0.74，肩寬少 3～5 cm、袖長少 2～3 cm（縫好、攤平前其實和版片差 0.3 內）。
    從沒有翻折的位置再縫，邊回到原長；彎曲剛度調大（RESEW_BEND），布形成平順的鼓起而不是再折成尖角。
    攤平用「方向統一過」的面（sew.orient_faces）：GarmentCode 的翻領、前襟三角形順序相反，
    不統一的話 folded_faces 會把領子誤判成尖角，每次都去抹平領子（襯衫、POLO 各 40～60 個假警報）。
    實測（2026-11-24，docs/驗證_尺寸）：肩寬差 z06 M −4.0 → −0.5、z04 M −4.9 → −2.8；胸寬、衣長變動 0.5 cm 內。"""
    fo, _ = orient_faces(f)
    edges = mesh_edges(f)
    rest = rest_lengths(box, edges, pickle.loads((folder / f"{name}_orig_lens.pickle").read_bytes()))
    x, info = unfold(x, fo, body=body_mesh, steps=3, max_ring=0)
    log, sq = [info], []
    best = None
    for r in range(RESEW_MAX_ROUNDS if squeeze else RESEW_ROUNDS):
        x = sew(x, edges, rest, body=body_mesh, **{**SEW, "bend": RESEW_BEND, "iterations": RESEW_ITERS})
        x, info = unfold(x, fo, body=body_mesh, steps=3, max_ring=0)
        log.append(info)
        if squeeze is None:
            continue
        s = float(squeeze(x))
        sq.append(round(s, 2))
        if r + 1 >= RESEW_ROUNDS and (best is None or s < best[0]):
            best = (s, x.copy())
        if r + 1 >= RESEW_ROUNDS and s <= SQUEEZE_OK:
            break
    if best is not None:
        x = best[1]
    return x, log, sq


NAMES = {"chest": "胸寬", "shoulder": "肩寬", "length": "衣長", "sleeve": "袖長", "hem": "下襬圍"}
CHEST_BELOW_ARMPIT = 2.5


def girth(pts):
    """一圈點（xz 平面）的周長，用凸包（平放量的是外圍，凹進去的皺褶不算）。"""
    return float(ConvexHull(pts).area) if len(pts) >= 3 else float("nan")


def measure_3d(x, f, seg, lm):
    """縫好的衣服（cm、腳底在 0、T-pose）量尺寸表的幾個欄位。seg：每個頂點的版片名稱；lm：spec_measure.landmarks()。"""
    seg = np.asarray(seg)
    torso = np.array(["torso" in s for s in seg])
    sleeve = np.array(["sleeve" in s or "cuff" in s for s in seg])
    out = {}
    # 腋下：袖子靠身體那一端最低的點
    near = sleeve & (np.abs(x[:, 0]) < np.abs(x[torso, 0]).max() + 4)
    armpit = x[near, 1].min() if near.any() else x[torso, 1].max() - 20
    y0 = armpit - CHEST_BELOW_ARMPIT
    band = torso & (np.abs(x[:, 1] - y0) < 1.0)
    out["chest"] = girth(x[band][:, [0, 2]]) / 2
    out["length"] = smeas.length(x, f, seg, lm)
    out["shoulder"] = smeas.shoulder(x, f, seg, lm)
    out["sleeve"] = smeas.sleeve(x, f, seg, lm)
    loops = igl.all_boundary_loop(f)
    hem = min(loops, key=lambda L: x[L, 1].mean())
    p = x[hem]
    out["hem"] = float(np.linalg.norm(p - np.roll(p, 1, 0), axis=1).sum())
    return out


def code(gid):
    """z06_長袖T → z06（輸出路徑只用英文代號）"""
    return gid.split("_")[0]


# 下襬羅紋（收緊）：GarmentCode 的腰帶（meta.wb＝StraightWB）直接縫在上衣下襬，腰帶比衣身短，縫的時候衣身被抽縐收進去。
# 高度 cm；ratio＝羅紋一圈 ÷ 衣服胸圍（帽 T 常見 0.85）；最少到身體臀圍，不然會被身體撐開。尺寸表的衣長包含羅紋。
HEM_RIB = {"z01_帽T": {"height_cm": 6.0, "ratio": 0.85}}


def template_for(gid, body):
    design = load_design(AI / TEMPLATES[gid])
    if design["meta"]["upper"]["v"] == "FittedShirt":  # 合身款不看衣長參數
        design["meta"]["upper"]["v"] = "Shirt"
    if gid in HEM_RIB:
        design["meta"]["wb"]["v"] = "StraightWB"
        design["_hem_rib"] = dict(HEM_RIB[gid])  # fit2 照胸寬算羅紋一圈多長（GarmentCode 不看這個鍵）
    with contextlib.redirect_stdout(io.StringIO()):
        fix_neck_depth(design, body, MetaGarment)
    return design


def table_value(col, v):
    """尺寸表的值轉成數字；「前50 後54」這種前後不同的衣長取後面（尺寸表的衣長從後領量）。"""
    if not isinstance(v, str):
        return float(v)
    if "後" in v:
        return float(v.split("後")[1].strip())
    return None


def hem_2d(pattern):
    """2D 版片下襬一圈：每片軀幹版片在最底下（高 0.5 cm）的寬度加總。"""
    total = 0.0
    for n, p in pattern["panels"].items():
        if "torso" in n:
            _, es = sp.sample_panel(p)
            ys = np.vstack(es)[:, 1]
            total += sp.width_at(es, ys.min() + 0.5)
    return total


DROP_SHOULDER_INCLS = (12.0, 14.0, 16.0, 18.0, 20.0)  # 落肩款打版用的肩斜，從平的開始試（身體原本約 22°，縫起來袖子會掉）；
# 太平 GarmentCode 畫不出寬肩，能用的最小值每件不同（z06 12°、帽 T 14°、Polo 16°，帽子、翻領會吃掉空間）


def max_shoulder(design, incl, sw_lo, sw_hi, extra=None):
    """這個肩斜下，打版用肩寬在 [sw_lo, sw_hi] 內畫得出來的最大值（二分搜尋）；sw_lo 都畫不出來回傳 None。
    extra：其他打版用身體的修改（例如 back_width）。"""
    def ok(sw):
        try:
            return sp.measure(sp.build(design, {**(extra or {}), "shoulder_w": sw, "shoulder_incl": incl}))["shoulder"]
        except AssertionError:
            return None
    if ok(sw_lo) is None:
        return None
    if (m := ok(sw_hi)) is not None:
        return sw_hi, m
    lo, hi = sw_lo, sw_hi
    while hi - lo > 0.5:
        mid = (lo + hi) / 2
        if ok(mid) is None:
            hi = mid
        else:
            lo = mid
    return lo, ok(lo)


def choose_drop(design, target_shoulder, sw0, base, extra=None):
    """落肩款：挑最平、又能把肩寬推到目標（推不到就挑推得最寬）的肩斜。回傳 (肩斜, 打版用肩寬, 2D 推得到的肩寬)。"""
    want = sw0 + (target_shoulder - base) + 2  # 多給 2 cm 搜尋空間
    best = None
    for incl in DROP_SHOULDER_INCLS:
        r = max_shoulder(design, incl, sw0, want, extra)
        if r is None:
            continue
        sw, reach = r
        if reach >= target_shoulder - 0.5:
            return incl, min(sw, sw0 + (target_shoulder - base)), reach
        if best is None or reach > best[2] + 1.0:
            best = (incl, sw, reach)
    return best if best else (None, sw0, None)
DROP_MARGIN_CM = 3.0  # 尺寸表肩寬比預設版片肩寬多這麼多，才當落肩款處理


def fit2(target, template, drop_class=False):
    """在 2D 版片上把胸寬、衣長、袖長、下襬圍、肩寬修到差 0.2 cm 內。
    drop_class：判定是落肩款時，上衣換成落肩版型 DropShoulderShirt（drop_shoulder.py，肩點可以比胸寬更外面）。
    下襬圍 ＝ 下襬外擴 shirt.flare × 胸圍方向的寬度，所以 flare 跟著胸寬一起修。
    肩寬：一般上衣沒有肩寬參數，肩點跟著「打版用的身體肩寬 shoulder_w」走，所以改打版用的尺寸檔（3D 身體不變）；
      比身體寬（落肩）時同時把肩斜放平到 DROP_SHOULDER_INCL、袖子角度改 0，不然縫起來袖子會掉到手臂下面
      （experiments/shoulder_try.py）；推到 GarmentCode 畫不出來（袖籠沒空間，約後片寬）就停在畫得出來的最大值。
      推不到時把打版用身體的 back_width 改成胸圍÷2：GarmentCode 的側縫偏後、後片比較窄，肩點最遠只到後片邊緣；
      側縫移到正側邊後片變寬，能多推 2.4～2.8 cm（z06 全部到尺寸表 0.5 內）。肩寬比胸寬還大的（z04）還是推不到。
    袖山抽皺：落肩 RUFFLE_DROP、一般肩 RUFFLE_REGULAR（AI 給的 1.11～1.18 會讓肩點附近被縫線拉皺）。
    回傳 (design, 量到, 參數, 打版用身體的修改)。"""
    design = copy.deepcopy(template)
    design["shirt"]["flare"]["v"] = 1.0
    design["sleeve"]["connect_ruffle"]["v"] = RUFFLE_REGULAR
    bust = sp.BASE_BODY.params["bust"]
    design["shirt"]["width"]["v"] = target["chest"] * 2 / bust
    if design.get("_hem_rib"):  # 下襬羅紋：一圈＝max(ratio × 衣服胸圍, 身體臀圍)；腰帶參數是身體腰圍、臀線長的倍數
        rib = design["_hem_rib"]
        circ = max(rib["ratio"] * target["chest"] * 2, sp.BASE_BODY.params["hips"])
        design["waistband"]["waist"]["v"] = circ / sp.BASE_BODY.params["waist"]
        design["waistband"]["width"]["v"] = rib["height_cm"] / sp.BASE_BODY.params["hips_line"]
        design["waistband"]["height"]["v"] = 0
    body_extra = {}
    if "shoulder" in target:
        sw0 = sp.BASE_BODY.params["shoulder_w"]
        base = sp.measure(sp.build(design, {}))["shoulder"]
        body_extra["shoulder_w"] = sw0 + (target["shoulder"] - base)
        if target["shoulder"] > base + DROP_MARGIN_CM:
            design["sleeve"]["sleeve_angle"]["v"] = 0
            design["sleeve"]["connect_ruffle"]["v"] = RUFFLE_DROP
            if drop_class:
                design["meta"]["upper"]["v"] = "DropShoulderShirt"
            incl, sw, reach = choose_drop(design, target["shoulder"], sw0, base)
            if not drop_class and reach is not None and reach < target["shoulder"] - 0.5:  # 推不到 → 側縫移到正側邊再試
                wide = {"back_width": bust / 2}
                r = choose_drop(design, target["shoulder"], sw0, base, wide)
                if r[2] is not None and r[2] > reach + 0.2:
                    incl, sw, reach = r
                    body_extra.update(wide)
            if incl is not None:
                body_extra["shoulder_incl"] = incl
                body_extra["shoulder_w"] = sw
    good_sw = sp.BASE_BODY.params["shoulder_w"]
    cap = None  # 肩寬推到畫不出來的位置後，記住畫得出來的最大打版用肩寬
    m = {}
    for _ in range(20):
        if "hem" in target:
            design["shirt"]["flare"]["v"] = target["hem"] / (design["shirt"]["width"]["v"] * bust)
        if cap is not None and "shoulder_w" in body_extra:
            body_extra["shoulder_w"] = min(body_extra["shoulder_w"], cap)
        try:
            pat = sp.build(design, body_extra)
        except AssertionError:
            if "shoulder_w" not in body_extra or body_extra["shoulder_w"] <= sp.BASE_BODY.params["shoulder_w"]:
                raise
            if body_extra["shoulder_w"] <= good_sw + 0.2:
                # 胸寬修過後，上次畫得出來的肩寬也畫不出來了 → 再往內退 1 cm
                cap = good_sw = body_extra["shoulder_w"] = body_extra["shoulder_w"] - 1.0
                continue
            # 肩推太外面畫不出來 → 退一半；退到離畫得出來的值很近時，就把它當上限
            body_extra["shoulder_w"] = (body_extra["shoulder_w"] + good_sw) / 2
            if body_extra["shoulder_w"] - good_sw < 0.3:
                cap = body_extra["shoulder_w"] = good_sw
            continue
        good_sw = body_extra.get("shoulder_w", good_sw)
        m = sp.measure(pat)
        m["hem"] = hem_2d(pat)
        diff = {k: m[k] - target[k] for k in target}
        shoulder_ok = abs(diff.get("shoulder", 0)) <= 0.5 or (cap is not None and diff["shoulder"] < 0)
        if all(abs(d) <= 0.2 for k, d in diff.items() if k != "shoulder") and shoulder_ok:
            break
        design["shirt"]["width"]["v"] *= target["chest"] / m["chest"]
        if "length" in target:
            design["shirt"]["length"]["v"] *= target["length"] / m["length"]
        if "sleeve" in target:
            design["sleeve"]["length"]["v"] *= target["sleeve"] / m["sleeve"]
        if "shoulder" in target:
            body_extra["shoulder_w"] -= diff["shoulder"]
    if "shoulder_w" in body_extra:
        body_extra["shoulder_w"] = good_sw
    params = {"shirt.width": design["shirt"]["width"]["v"], "shirt.length": design["shirt"]["length"]["v"],
              "shirt.flare": design["shirt"]["flare"]["v"], "sleeve.length": design["sleeve"]["length"]["v"],
              **{f"打版用 {k}": v for k, v in body_extra.items()}}
    return design, m, params, body_extra


def pack_mesh(x, f):
    """網頁用：頂點量化到 mm（int16）、面（uint16），都轉成 base64。"""
    q = np.round(x * 10).astype(np.int16)
    return {"v": base64.b64encode(q.tobytes()).decode(), "f": base64.b64encode(f.astype(np.uint16).tobytes()).decode(),
            "nv": int(len(x)), "nf": int(len(f))}


def run(gid, body_obj, body_yaml, drop=None, sizes=None):
    """照尺寸表做這件的每個尺碼：在 2D 版片上對準尺寸表，縫一次，量 3D 記下來（不再用 3D 的差回頭修版片）。
    drop：落肩款什麼時候換成落肩版型 DropShoulderShirt
      None ＝ 不換（2026-11-25 以前的做法）
      "a"  ＝ 先用 Shirt，肩寬推不到尺寸表（差超過 0.5）才換
      "b"  ＝ 判定是落肩款就換
    sizes：只做這些尺碼（None＝全部）"""
    info = json.loads((DATA / gid / "尺寸.json").read_text(encoding="utf-8"))
    sp.BASE_BODY = BodyParameters(str(body_yaml))
    bv, bf = igl.read_triangle_mesh(str(body_obj))
    body_mesh = (bv * 100, bf)
    template = template_for(gid, sp.BASE_BODY)
    keys = dict(sp.KEYS, **{"すそ周り": "hem"})
    for size, vals in info["sizes"].items():
        if sizes and size not in sizes:
            continue
        dst = OUT / code(gid) / size
        if (dst / "量測.json").exists():
            print(f"[{gid} {size}] 已經做過", flush=True)
            continue
        t = time.time()
        table = {keys[c]: table_value(c, v) for c, v in zip(info["cols"], vals) if c in keys}
        table = {k: v for k, v in table.items() if v is not None}
        target = {k: v for k, v in table.items() if k in ("chest", "length", "sleeve", "hem", "shoulder")}
        name = f"specb_{gid.split('_')[0]}_{size}"
        bust = sp.BASE_BODY.params["bust"]
        dst.mkdir(parents=True, exist_ok=True)
        skip = dst / "跳過.json"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                design, m2d, params, body_extra = fit2(target, template, drop_class=(drop == "b"))
                if drop == "a" and "shoulder" in target and m2d["shoulder"] < target["shoulder"] - 0.5:
                    design, m2d, params, body_extra = fit2(target, template, drop_class=True)
        except AssertionError:
            # 照尺寸表畫不出版片（衣服太小：衣身窄到袖籠塞不進肩角，GarmentCode 報錯）→ 沒有結果可用，
            # 記下來跳過這個尺碼，繼續做下一個。只寫 跳過.json、不寫 量測.json，出圖、匯出、總表都會自動略過；
            # 下次重跑會再試一次（例如換了打版身體之後）。
            skip.write_text(json.dumps({
                "衣服": gid, "尺碼": size, "尺寸表": table,
                "原因": "照尺寸表畫不出版片（GarmentCode 報錯，衣服可能比打版身體能畫的最小值還小）",
                "打版身體半胸圍": round(bust / 2, 1), "秒": round(time.time() - t),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{gid} {size}] 跳過：照尺寸表畫不出版片，記在 {skip}", flush=True)
            continue
        yml = body_yaml
        if body_extra:  # 打版用的尺寸檔改肩寬、肩斜、後片寬（3D 身體不變）
            b = yaml.safe_load(Path(body_yaml).read_text())
            b["body"].update({k: float(v) for k, v in body_extra.items()})
            yml = OUT / "_bodies" / f"{code(gid)}_{size}.yaml"
            yml.parent.mkdir(parents=True, exist_ok=True)
            yml.write_text(yaml.safe_dump(b), encoding="utf-8")
        res, _ = pick_resolution(design, name, yml)
        folder, x, f, box, _ = build(design, name, yml, body_mesh, resolution=res)
        seg = (folder / f"{name}_sim_segmentation.txt").read_text().splitlines()
        lm = smeas.landmarks(folder, name)
        squeeze = None
        if "shoulder" in target:
            def squeeze(xx):
                return smeas.shoulder_inset(xx, f, seg, lm, folder, name) - smeas.shoulder(xx, f, seg, lm)
        x, unf, sq = sew_finish(x, f, box, folder, name, body_mesh, squeeze)
        m3d = measure_3d(x, f, seg, lm)
        m3d["reach"] = m3d["shoulder"] / 2 + m3d["sleeve"]
        skip.unlink(missing_ok=True)  # 之前跳過、這次做得出來
        (dst / "segmentation.txt").write_text("\n".join(seg), encoding="utf-8")  # 每個頂點屬於哪片版片（重量用）
        igl.write_triangle_mesh(str(dst / f"{code(gid)}_{size}.obj"), x, f)
        (dst / "design.yaml").write_text(yaml.safe_dump(json.loads(json.dumps({"design": design}, default=float)),
                                                        allow_unicode=True), encoding="utf-8")
        rec = {"衣服": gid, "尺碼": size, "款式來源": TEMPLATES[gid], "尺寸表": table,
               "版片量到": {k: round(v, 1) for k, v in m2d.items()},
               "縫好量到": {k: round(v, 1) for k, v in m3d.items()}, "參數": {k: round(v, 3) for k, v in params.items()},
               "上衣版型": design["meta"]["upper"]["v"], "再縫每次的肩角壓縮": sq,
               "照尺寸表的欄位": sorted(target), "量測點": lm, "頂點數": int(len(x)), "頂點間距cm": round(res, 2), "尖角攤平": unf,
               "秒": round(time.time() - t)}
        if "shoulder" in target and abs(m2d["shoulder"] - target["shoulder"]) > 1.0:
            rec["2D 推不到"] = f"肩寬最多推到 {m2d['shoulder']:.1f}（尺寸表 {target['shoulder']}）"
        if "chest" in target and target["chest"] * 2 < bust:
            rec["比身體小"] = f"尺寸表胸圍 {target['chest'] * 2:.0f} < 打版身體胸圍 {bust:.0f}，穿上去一定被撐開"
        if "shoulder" in target:
            ins = smeas.shoulder_inset(x, f, seg, lm, folder, name)
            rec["肩角被壓(診斷)"] = round(ins - m3d["shoulder"], 1)
        (dst / "量測.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{gid} {size}] 完成 {rec['秒']} 秒｜" + "  ".join(
            f"{NAMES[k]} 3D {m3d[k]:.1f}／版 {m2d[k]:.1f}／表 {table[k]}" for k in table), flush=True)


def remeasure():
    """量法改了時用：讀縫好的網格、這個尺碼自己的分區檔和量測點，重算「縫好量到」，不用重縫。
    沒存量測點的舊結果，從縫合資料夾重找（資料夾可能已被之後的縫合蓋掉：頂點數對不上就略過）。"""
    for d in sorted(OUT.glob("*/*/量測.json")):
        rec = json.loads(d.read_text(encoding="utf-8"))
        name = f"specb_{code(rec['衣服'])}_{rec['尺碼']}"
        x, f = igl.read_triangle_mesh(str(d.parent / f"{code(rec['衣服'])}_{rec['尺碼']}.obj"))
        own = d.parent / "segmentation.txt"
        seg_file = own if own.exists() else sp.dm.OUT_ROOT / name / f"{name}_sim_segmentation.txt"
        seg = seg_file.read_text(encoding="utf-8").splitlines()
        folder = sp.dm.OUT_ROOT / name
        if len(seg) != len(x) or ("量測點" not in rec and len(folder.joinpath(f"{name}_sim_segmentation.txt")
                                                          .read_text().splitlines()) != len(x)):
            print(f"[{rec['衣服']} {rec['尺碼']}] 分區檔或縫合資料夾和網格對不上，略過（要重縫）", flush=True)
            continue
        lm = rec.get("量測點") or smeas.landmarks(folder, name)
        m3d = measure_3d(x, f, seg, lm)
        m3d["reach"] = m3d["shoulder"] / 2 + m3d["sleeve"]
        rec["縫好量到"] = {k: round(v, 1) for k, v in m3d.items()}
        rec["量測點"] = lm
        d.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{rec['衣服']} {rec['尺碼']}] " + "  ".join(f"{NAMES.get(k, k)} {m3d[k]:.1f}/{rec['尺寸表'].get(k, '-')}"
                                                         for k in ("chest", "shoulder", "length", "sleeve", "hem")), flush=True)


def summary():
    """把每件、每個尺碼的量測合成一份（網頁用），附網格。"""
    items = []
    for gid in TEMPLATES:
        info = json.loads((DATA / gid / "尺寸.json").read_text(encoding="utf-8"))
        for size in info["sizes"]:
            d = OUT / code(gid) / size
            if not (d / "量測.json").exists():
                continue
            rec = json.loads((d / "量測.json").read_text(encoding="utf-8"))
            x, f = igl.read_triangle_mesh(str(d / f"{code(gid)}_{size}.obj"))
            rec["mesh"] = pack_mesh(x, f)
            items.append(rec)
        meta = {k: info.get(k) for k in ("name", "brand", "type", "features", "material", "url")}
        for r in items:
            if r["衣服"] == gid:
                r["商品"] = meta
                r["款式來源"] = TEMPLATES[gid]
    (OUT / "總表.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} 件 → {OUT / '總表.json'}")


def main(*args):
    if "--summary" in args:
        return summary()
    if "--remeasure" in args:
        return remeasure()
    global OUT
    args = list(args)
    opt = {}
    for k in ("--drop", "--out", "--sizes"):  # --drop a|b、--out 輸出資料夾名（output/spec_to_pattern/ 下）、--sizes S,M
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            del args[i:i + 2]
    if "--out" in opt:
        OUT = OUT.parent / opt["--out"]
    sizes = opt["--sizes"].split(",") if "--sizes" in opt else None
    sp.dm.ensure_system_json()
    body_obj, body_yaml, _ = prepare_smplx_body("neutral")
    for gid in (args or TEMPLATES):
        run(gid, body_obj, body_yaml, drop=opt.get("--drop"), sizes=sizes)


if __name__ == "__main__":
    main(*sys.argv[1:])
