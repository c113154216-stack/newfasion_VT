# -*- coding: utf-8 -*-
"""
SMPL-X 人體模型導入與對齊。

真正的 SMPL-X 需要官方授權的模型權重檔（.npz），必須本人到
https://smpl-x.is.tue.mpg.de/ 註冊、同意授權條款後才能下載，無法由程式自動
取得，所以權重就緒前這裡的 load_smplx_body() 會丟出清楚的錯誤訊息。

在那之前，先用 build_proxy_mannequin() 產生一個純幾何、跟 geometry_engine.py
的公版短袖上衣同一套「輪廓內插 + lathe」手法做出來的替代人偶，可以立刻測試
「對齊」這件事的邏輯本身對不對，之後把 SMPL-X 權重放進 assets/smplx/，
只要把呼叫換成 load_smplx_body() 就好，對齊 / 後續流程不用改。
"""

import os
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
SMPLX_MODEL_DIR = os.path.join(ASSETS_DIR, 'smplx')  # 實際權重檔 (SMPLX_NEUTRAL.npz 等) 放這裡

# 人體比例基準（以身高為 1.0 的相對高度，腳底=0）：概略常見人體工學比例，
# 用來讓 proxy 人偶「胸/腰/臀/肩」落在合理位置，也用來跟服飾網格對齊校準。
#
# shirt_hem 原本是憑經驗猜的 0.56，實測發現對齊後衣領會貼到下巴附近，明顯偏高。
# 改用 SMPL-X 真正的關節座標校正：中性體型 (natural pose) 量出來
# 左右 hip 關節平均 y = -0.4498，身高範圍 -1.3018 ~ 0.4191（1.7209m），
# 換算成相對比例 = (-0.4498 - (-1.3018)) / 1.7209 ≈ 0.495——比原本猜的 0.56
# 低了快 7 個百分點（約 11cm），這才是這件短袖上衣下擺該落在的髖部高度。
HEIGHT_RATIOS = {
    'head_top':  1.00,
    'shoulder':  0.82,
    'chest':     0.74,   # 胸圍最寬處，約腋下高度
    'shirt_hem': 0.495,  # 短袖上衣下擺大約落點，改用 SMPL-X 髖關節實測值校正
    'waist':     0.60,
    'hip':       0.52,
    'crotch':    0.48,
}


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grid_faces(row_segments, col_segments):
    cols = col_segments + 1
    ai, aj = np.meshgrid(np.arange(row_segments), np.arange(col_segments), indexing='ij')
    a = (ai * cols + aj).ravel()
    b = a + 1
    c = a + cols
    d = c + 1
    tri1 = np.stack([a, c, d], axis=1)
    tri2 = np.stack([a, d, b], axis=1)
    return np.vstack([tri1, tri2])


def build_proxy_mannequin(height_cm=170.0, bust_cm=90.0, waist_cm=75.0, hip_cm=95.0,
                           radial_segments=32, height_segments=48):
    """
    純幾何的替代人偶：沿身高方向堆疊橢圓截面，半徑依 HEIGHT_RATIOS 定義的
    胸/腰/臀分段用 smoothstep 內插（周長轉半徑：r = circumference / 2π）。

    回傳單位跟服飾網格一致（公尺），Y 軸為身高方向，Y=0 是腳底。
    """
    height_m = height_cm / 100.0
    bust_r = (bust_cm / (2 * np.pi)) / 100.0
    waist_r = (waist_cm / (2 * np.pi)) / 100.0
    hip_r = (hip_cm / (2 * np.pi)) / 100.0
    shoulder_r = bust_r * 1.08  # 肩寬略大於胸圍半徑，經驗值

    keypoints = [
        (0.00,                         hip_r * 0.55),   # 腳踝
        (HEIGHT_RATIOS['crotch'] - 0.05, hip_r * 0.55),
        (HEIGHT_RATIOS['hip'],         hip_r),
        (HEIGHT_RATIOS['waist'],       waist_r),
        (HEIGHT_RATIOS['chest'],       bust_r),
        (HEIGHT_RATIOS['shoulder'],    shoulder_r),
        (HEIGHT_RATIOS['head_top'] - 0.06, bust_r * 0.35),  # 頸部收窄
        (HEIGHT_RATIOS['head_top'],    bust_r * 0.35),
    ]

    t = np.arange(height_segments + 1) / height_segments
    theta = (np.arange(radial_segments + 1) / radial_segments) * 2.0 * np.pi
    T, THETA = np.meshgrid(t, theta, indexing='ij')

    keys = np.array([k[0] for k in keypoints])
    radii = np.array([k[1] for k in keypoints])
    r = np.full(T.shape, radii[-1])
    r[T <= keys[0]] = radii[0]
    for i in range(len(keys) - 1):
        mask = (T > keys[i]) & (T <= keys[i + 1])
        local_t = _smoothstep(keys[i], keys[i + 1], T[mask])
        r[mask] = radii[i] + (radii[i + 1] - radii[i]) * local_t

    X = r * np.sin(THETA)
    Z = r * np.cos(THETA) * 0.75  # 前後扁一點，比圓柱更像人體橢圓截面
    Y = T * height_m

    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = _grid_faces(height_segments, radial_segments)
    return vertices, faces


# 手臂自然垂放的角度（相對 SMPL-X 標準 T-pose 往下轉的弧度）。
#
# 走了一段冤枉路才搞懂正確做法：一開始把服飾網格當成「不會動的固定物」，
# 想辦法轉動人體手臂去遷就袖子的方向，先後試過 1.5 rad（穿模檢查數字最漂亮
# 但視覺完全錯，手臂縮回身側、袖子空蕩蕩地伸在外面）跟 0.35 rad（勉強對上
# 但手臂還是沒真的穿進袖子裡）。
#
# 真正的原因是：MGN 的服飾網格本來就是「SMPL 標準 T-pose 姿勢下」的資料，
# 跟 SMPL-X 的 canonical 座標系是同一套（實測服飾下擺 Y=-0.360 對上骨盆關節
# Y=-0.351、袖口 X=±0.40 剛好落在肩 ±0.16 與手肘 ±0.42 之間，X/Y 完全對齊，
# 只有 Z 差 2.6cm 是衣服前襟自然的寬鬆量）。既然是同一套座標系，服飾根本
# 不需要平移對齊，該做的是讓服飾「跟著骨架一起動」——也就是 pose_garment()
# 做的蒙皮權重轉移。轉好之後手臂想擺什麼角度，袖子都會自己跟上去。
NATURAL_SHOULDER_ANGLE = 0.5


def _natural_body_pose(shoulder_angle=NATURAL_SHOULDER_ANGLE, elbow_bend=0.0):
    import torch
    body_pose = torch.zeros((1, 21, 3))
    body_pose[0, 15, 2] = -shoulder_angle  # 左肩
    body_pose[0, 16, 2] = shoulder_angle   # 右肩
    body_pose[0, 17, 2] = -elbow_bend      # 左手肘
    body_pose[0, 18, 2] = elbow_bend       # 右手肘
    return body_pose.reshape(1, -1)


_BODY_FIT_REF_CACHE = {}


def _body_fit_reference(betas, gender, model_dir):
    """
    算出 fit_garment_to_body() 需要的「身體側」基準值：肩膀表面高度、髖關節
    高度、軀幹最寬處的高度區間跟半徑。這些只跟體型（betas/gender）有關，
    跟服飾本身的尺寸滑桿無關，所以可以照 betas 快取——注意快取的是「身體
    量出來的基準值」，不是套用在服飾上的最終結果，這樣不管服飾被滑桿調成
    什麼形狀，都一定會用最新的頂點座標去算，不會拿到過期結果。
    """
    import torch

    key = (gender, model_dir, tuple(np.asarray(betas).ravel().round(4)) if betas is not None else None)
    if key in _BODY_FIT_REF_CACHE:
        return _BODY_FIT_REF_CACHE[key]

    model = get_smplx_model(gender=gender, model_dir=model_dir)
    if betas is None:
        betas_t = torch.zeros((1, model.num_betas))
    else:
        betas_t = torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1)

    out = model(betas=betas_t, return_verts=True)          # canonical T-pose
    body = out.vertices.detach().cpu().numpy().squeeze(0)
    J = out.joints.detach().cpu().numpy().squeeze(0)

    # 肩膀表面高度：取肩區 (0.10 < |x| < 0.18) 的最高頂點。這個 x 範圍是實測
    # 挑出來的——再往內會抓到頭/耳朵（|x|<0.14 時最高點會跳到 0.34）。
    shoulder_band = (np.abs(body[:, 0]) > 0.10) & (np.abs(body[:, 0]) < 0.18)
    shoulder_top_y = float(body[shoulder_band, 1].max())
    hip_y = float((J[1, 1] + J[2, 1]) / 2.0)

    ref = {'body': body, 'shoulder_top_y': shoulder_top_y, 'hip_y': hip_y}
    _BODY_FIT_REF_CACHE[key] = ref
    return ref


def fit_garment_to_body(garment_vertices, betas=None, gender='neutral',
                         model_dir=SMPLX_MODEL_DIR, ease=0.015, ref_y_extent=None):
    """
    把外部服飾網格正規化成貼合 SMPL-X 標準體型的「公版」尺寸。

    為什麼需要這一步：MGN 資料集的服飾是套在「某個真實受試者」身上掃描下來的，
    那個人比 SMPL-X 的中性平均體型高大不少。實測原始網格「領口→下擺」跨距
    0.683m，但 SMPL-X 平均體型「頸→髖」只有 0.558m，服飾整整大了 22.6%，
    直接疊上去領口會跑到下巴以上、整件衣服看起來浮在身體上方。

    垂直與水平分開處理，因為兩者的合身標準不一樣：
      - 垂直（Y）：衣服最高點對齊「肩膀表面」、下擺對齊髖關節，決定衣長。
        注意要用肩膀表面高度（實測 y=+0.1486）而不是頸關節（y=+0.1077）——
        頸關節其實比肩膀表面還低 4cm，拿它當基準會讓整件衣服往下掉，
        肩膀會裸露在外面。
      - 水平（X/Z）：讓胸圍比身體再寬 ease 公尺（預設 1.5cm 的鬆份），
        衣服本來就該比身體寬鬆，如果跟垂直用同一個等比例縮放，胸圍會偏緊。

    這個函式本身刻意不快取結果：它是每次都要重新套用在「當下傳進來的」
    garment_vertices 上（滑桿調完的最新形狀），快取結果的話，改肩寬/胸圍
    滑桿只會改變頂點座標、不會改變頂點數量，快取 key 抓不到差異，就會一直
    拿到第一次呼叫時的舊結果——這是真的踩過的 bug，之前拿頂點數量當 key
    快取整個函式的回傳值，導致服飾滑桿在 SMPL-X 模式下完全沒有作用。
    真正耗時的部分（SMPL-X forward pass 量身體基準值）拆到
    _body_fit_reference() 用 betas 當 key 快取，這部分才是安全可以重複使用的。

    ref_y_extent：算垂直縮放比例時要拿來當分母的「衣長跨距」。如果不給，
    預設用 garment_vertices 自己的實際 Y 跨距——但這樣會把「衣長」滑桿的
    效果完全吃掉：不管使用者把衣服拉多長/多短，這一步都會把它硬縮放成
    「肩膀表面到髖關節」那個固定跨距，衣長滑桿等於白調。正確做法是傳入
    「衣長=100（預設值）時的公版跨距」當分母，這樣縮放比例是固定的，
    衣服被滑桿拉長多少，套用同一個比例後下擺就會實際往下移動多少。
    """
    ref = _body_fit_reference(betas, gender, model_dir)
    body = ref['body']
    shoulder_top_y, hip_y = ref['shoulder_top_y'], ref['hip_y']

    g = np.asarray(garment_vertices, dtype=np.float64).copy()

    # --- 垂直：衣服頂端對齊肩膀表面，縮放比例固定，讓衣長滑桿仍然有效 ---
    denom = ref_y_extent if ref_y_extent is not None else (g[:, 1].max() - g[:, 1].min())
    y_scale = (shoulder_top_y - hip_y) / denom
    g[:, 1] *= y_scale
    g[:, 1] += shoulder_top_y - g[:, 1].max()

    # --- 水平：讓胸圍比身體多留 ease 的鬆份 ---
    # 胸圍位置取「服飾涵蓋範圍內，身體軀幹最寬的那一段」，不寫死高度。
    torso = np.abs(body[:, 0]) < 0.25
    band = torso & (body[:, 1] > g[:, 1].min()) & (body[:, 1] < g[:, 1].max())
    ys = body[band, 1]
    bins = np.linspace(ys.min(), ys.max(), 12)
    best_r, best_lo, best_hi = 0.0, bins[0], bins[1]
    for lo, hi in zip(bins, bins[1:]):
        sel = band & (body[:, 1] >= lo) & (body[:, 1] < hi)
        if sel.sum() < 10:
            continue
        r = np.percentile(np.hypot(body[sel, 0], body[sel, 2]), 90)
        if r > best_r:
            best_r, best_lo, best_hi = r, lo, hi

    gm = (np.abs(g[:, 0]) < 0.25) & (g[:, 1] >= best_lo) & (g[:, 1] < best_hi)
    if gm.sum() >= 10:
        garment_r = np.percentile(np.hypot(g[gm, 0], g[gm, 2]), 90)
        if garment_r > 1e-6:
            xz_scale = (best_r + ease) / garment_r
            g[:, 0] *= xz_scale
            g[:, 2] *= xz_scale

    return g


_GARMENT_SKIN_CACHE = {}


def _garment_skinning_weights(garment_vertices, model, betas, gender='neutral'):
    """
    服飾網格本身沒有骨架資訊，這裡用「最近鄰借權重」的方式補上：對每個服飾
    頂點，找出 canonical（T-pose）人體上離它最近的頂點，直接借用那個頂點的
    SMPL-X 蒙皮權重。MGN 官方的 dress_SMPL.py 也是用類似的頂點對應概念。

    回傳 (服飾頂點數, 55) 的權重矩陣。結果會依服飾頂點數與 betas 快取，
    因為建 KD-tree + 查詢在 7702 個頂點時要花幾百毫秒，不適合每次請求重算。

    快取 key 一定要包含 gender：不同性別的模型是完全獨立的權重檔，betas
    數值本身沒有跨性別的意義（例如兩邊都剛好是全 0），只用 betas 當 key
    在多性別上線後會拿錯性別的蒙皮權重去套。
    """
    import torch
    from smplx.lbs import blend_shapes, vertices2joints
    from scipy.spatial import cKDTree

    key = (garment_vertices.shape[0], gender, tuple(np.asarray(betas).ravel().round(4)))
    if key in _GARMENT_SKIN_CACHE:
        return _GARMENT_SKIN_CACHE[key]

    # canonical 人體（只套用體型 betas、不套用任何姿勢）
    v_shaped = model.v_template.unsqueeze(0) + blend_shapes(betas, model.shapedirs)
    v_can = v_shaped[0].detach().cpu().numpy()

    _, idx = cKDTree(v_can).query(garment_vertices)
    W = model.lbs_weights[idx]  # (n_garment, 55)

    J = vertices2joints(model.J_regressor, v_shaped)
    _GARMENT_SKIN_CACHE[key] = (W, J)
    return W, J


def pose_garment(garment_vertices, betas=None, gender='neutral',
                  model_dir=SMPLX_MODEL_DIR, pose='natural'):
    """
    用 SMPL-X 的骨架帶動服飾網格一起變形（Linear Blend Skinning）。

    服飾在 canonical T-pose 下本來就跟人體對齊，只要對服飾頂點套用跟人體
    完全相同的關節變換矩陣，人體擺什麼姿勢，衣服就會跟著擺到對的位置，
    袖子自然會包住手臂，不需要再手動去猜平移量或轉手臂角度去遷就它。
    """
    import torch
    from smplx.lbs import batch_rodrigues, batch_rigid_transform

    model = get_smplx_model(gender=gender, model_dir=model_dir)
    if betas is None:
        betas = torch.zeros((1, model.num_betas))
    else:
        betas = torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1)

    garment_vertices = np.asarray(garment_vertices, dtype=np.float64)
    W, J = _garment_skinning_weights(garment_vertices, model, betas, gender=gender)

    body_pose = _resolve_body_pose(pose)

    # 湊出 SMPL-X 完整的 55 個關節姿勢：global(1) + body(21) + jaw/eyes(3) + hands(30)
    n_extra = 55 - 1 - model.NUM_BODY_JOINTS
    full_pose = torch.cat([
        torch.zeros((1, 3)),                       # global_orient
        body_pose.reshape(1, -1),                  # body_pose (21 joints)
        torch.zeros((1, n_extra * 3)),             # jaw / eyes / hands 都維持預設
    ], dim=1)

    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
    _, A = batch_rigid_transform(rot_mats, J, model.parents)  # A = 各關節的相對變換矩陣

    # 每個頂點的變換 = 各關節變換依蒙皮權重線性混合（LBS 的核心）
    T = torch.matmul(W.unsqueeze(0), A.view(1, A.shape[1], 16)).view(1, -1, 4, 4)

    g = torch.as_tensor(garment_vertices, dtype=torch.float32)
    g_homo = torch.cat([g, torch.ones((g.shape[0], 1))], dim=1)
    posed = torch.matmul(T, g_homo.unsqueeze(0).unsqueeze(-1))[0, :, :3, 0]
    return posed.detach().cpu().numpy().astype(np.float64)


# 專門給「量身體尺寸」用的姿勢：雙臂放到接近完全垂直（1.45 rad），跟
# NATURAL_SHOULDER_ANGLE（0.5 rad，是配合服飾實際垂墜角度調的）不是同一個
# 用途。量胸圍/腰圍/臀圍需要手臂完全讓開軀幹兩側，不能用 T-pose（手臂水平
# 伸直，跟胸圍同高，量到的會是整條手臂的寬度）也不能用 NATURAL_SHOULDER_ANGLE
# （手臂還沒垂到底，前臂會斜切過腰、髖的高度污染量測），實測 1.45 rad 時
# 手臂已經貼近身體兩側但還沒真的貼到軀幹表面，胸/腰/臀高度的橫向延伸乾淨
# 不含手臂（x 範圍都收在 ±0.20 內，沒有像 0.5～1.2 rad 那樣手臂在旁邊拖一截）。
MEASURE_ARM_ANGLE = 1.45


def _measure_body_pose():
    import torch
    body_pose = torch.zeros((1, 21, 3))
    body_pose[0, 15, 2] = -MEASURE_ARM_ANGLE
    body_pose[0, 16, 2] = MEASURE_ARM_ANGLE
    return body_pose.reshape(1, -1)


def _band_girth_cm(vertices, y_ratio, y_min, height_m, x_limit=0.25, band_frac=0.015, pct=90):
    """
    量體型網格在某個高度比例處的圍度（公分）。

    做法：在目標高度附近切一個薄片（±band_frac×身高），只保留貼近軀幹中軸
    的頂點（|x|<x_limit，濾掉手臂），取 x／z 方向的 pct 百分位數當橢圓的
    半寬／半深，再用 Ramanujan 橢圓周長近似公式換算成圍度。之所以不是直接
    「2π×半徑」，是因為軀幹截面是扁橢圓不是正圓，實測直接用 2π×半徑算出來
    的胸圍會誇大到 125cm 以上（把最遠點當成整圈的半徑），橢圓公式量出來的
    平均體型胸圍 92cm、身高 172cm，比較貼近真實人體比例。
    """
    target_y = y_min + y_ratio * height_m
    band = band_frac * height_m
    mask = (np.abs(vertices[:, 1] - target_y) < band) & (np.abs(vertices[:, 0]) < x_limit)
    if mask.sum() < 5:
        return None
    a = np.percentile(np.abs(vertices[mask, 0]), pct)
    b = np.percentile(np.abs(vertices[mask, 2]), pct)
    h = ((a - b) / (a + b)) ** 2
    circumference = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    return float(circumference * 100)


def measure_body_cm(betas, gender='neutral', model_dir=SMPLX_MODEL_DIR):
    """量出一組 betas 對應的實際身高／胸圍／腰圍／臀圍（公分），單位跟前端滑桿一致。"""
    import torch

    model = get_smplx_model(gender=gender, model_dir=model_dir)
    betas_t = torch.zeros((1, model.num_betas))
    betas_arr = np.asarray(betas).ravel()
    betas_t[0, :len(betas_arr)] = torch.as_tensor(betas_arr, dtype=torch.float32)

    out = model(betas=betas_t, body_pose=_measure_body_pose(), return_verts=True)
    v = out.vertices.detach().cpu().numpy().squeeze(0)
    y = v[:, 1]
    y_min, height_m = float(y.min()), float(y.max() - y.min())

    return {
        'height_cm': height_m * 100,
        'bust_cm': _band_girth_cm(v, HEIGHT_RATIOS['chest'], y_min, height_m),
        'waist_cm': _band_girth_cm(v, HEIGHT_RATIOS['waist'], y_min, height_m),
        'hip_cm': _band_girth_cm(v, HEIGHT_RATIOS['hip'], y_min, height_m),
    }


_BETAS_FIT_CACHE = {}
_NUM_SHAPE_BETAS = 10  # 目前用的 SMPL-X 權重檔只有 10 個 shape 分量（model.num_betas）


def fit_betas_to_measurements(height_cm=170.0, bust_cm=90.0, waist_cm=75.0, hip_cm=95.0,
                               gender='neutral', model_dir=SMPLX_MODEL_DIR, iters=25):
    """
    幫使用者輸入的身高/胸圍/腰圍/臀圍（公分）反推一組 SMPL-X betas，讓
    measure_body_cm() 量出來的結果盡量貼近這四個目標值。

    做法：高斯牛頓法，每次迭代對目前的 betas 做一次「每個分量各自微擾一次」
    的有限差分求 Jacobian，再解一個加了 ridge 正則化的線性方程組更新 betas
    （正則化是為了避免解跑到不合理的極端體型、也讓方程組在 betas 維度比量測
    目標數量多時仍有唯一解）。scipy.optimize.least_squares 在這個問題上試過
    直接用，結果在 betas 起始值剛好是 0 時，數值微分的預設步長太小、在
    float32 精度下量出來的梯度是 0，一開始就誤判成「已經是最佳解」直接
    停止——所以改成自己寫這個固定步長（0.08）的版本，步長夠大不會被浮點
    誤差蓋掉，也可以自己控制疊代次數換取速度。

    已知限制：身高/胸圍/臀圍通常能收斂到跟目標差在 1~3cm 內，但「腰圍」在
    目標跟胸圍/臀圍差距很大（很明顯的沙漏型身材）時常常對不準（實測誤差
    可以到 10~20cm）——這個 SMPL-X 權重檔只有 10 個 shape 分量，量測的四個
    指標又高度相關（改身高/胸圍多少都會牽動腰圍/臀圍），10 個分量的 PCA
    空間沒辦法完全獨立控制这四個數字，這是形狀空間本身自由度不夠，不是
    程式邏輯的錯，目前先接受這個誤差，之後如果要更準，可能要考慮换成
    SHAPY 那套專門做「量測值->betas」迴歸的預訓練模型。

    結果依 (身高,胸圍,腰圍,臀圍,gender) 四捨五入到整數公分後快取——單次疊代
    要跑 25 次 forward pass，實測約 2~4 秒，同一組數值（例如使用者放開滑桿
    停在同一個整數上）不需要重算。另外加了提早停止：目標很接近平均體型
    （betas=0 附近，例如使用者只是把滑桿拉回預設值）時，通常 3~5 次疊代
    誤差就已經收斂到 1cm 以內，不需要真的跑完整整 25 次，這種常見情況下
    可以快很多。
    """
    key = (round(height_cm), round(bust_cm), round(waist_cm), round(hip_cm), gender)
    if key in _BETAS_FIT_CACHE:
        return _BETAS_FIT_CACHE[key]

    target = np.array([height_cm, bust_cm, waist_cm, hip_cm], dtype=np.float64)

    def measure_vec(betas_vec):
        m = measure_body_cm(betas_vec, gender=gender, model_dir=model_dir)
        return np.array([m['height_cm'], m['bust_cm'], m['waist_cm'], m['hip_cm']])

    step, ridge, clip, tol = 0.08, 0.3, 5.0, 1.0
    x = np.zeros(_NUM_SHAPE_BETAS)
    for _ in range(iters):
        f0 = measure_vec(x) - target
        if np.max(np.abs(f0)) < tol:
            break
        J = np.zeros((4, _NUM_SHAPE_BETAS))
        for j in range(_NUM_SHAPE_BETAS):
            xp = x.copy()
            xp[j] += step
            J[:, j] = (measure_vec(xp) - target - f0) / step
        A = J.T @ J + ridge * np.eye(_NUM_SHAPE_BETAS)
        b = J.T @ f0 + ridge * x
        x = np.clip(x - np.linalg.solve(A, b), -clip, clip)

    _BETAS_FIT_CACHE[key] = x
    return x


def smplx_body_from_measurements(height_cm=170.0, bust_cm=90.0, waist_cm=75.0, hip_cm=95.0,
                                  gender='neutral', model_dir=SMPLX_MODEL_DIR, pose='natural'):
    """
    高階入口：輸入使用者的身高/胸圍/腰圍/臀圍（公分），輸出對應體型、指定
    姿勢（預設 'natural'，手臂自然垂放方便跟服飾比對）的 SMPL-X 頂點與面，
    取代先前 build_body_smplx() 固定用 betas=None（平均體型）、身體滑桿完全
    沒作用的做法。回傳 (vertices, faces, betas)，betas 一併回傳是因為
    build_combo() 還要拿同一組 betas 去對服飾做 fit_garment_to_body() /
    pose_garment()，人體跟服飾的體型要用同一份 betas 才會真的合身。
    """
    betas = fit_betas_to_measurements(height_cm, bust_cm, waist_cm, hip_cm,
                                       gender=gender, model_dir=model_dir)
    vertices, faces = load_smplx_body(betas=betas, gender=gender, model_dir=model_dir, pose=pose)
    return vertices, faces, betas


# 給前端選的三種標準展示姿勢。跟服飾對齊用的 NATURAL_SHOULDER_ANGLE
# （0.5 rad）是兩回事——那個是蒙皮轉移剛做完、還在確認「服飾會不會跟著
# 骨架動」時，拿短袖實際垂墜角度校出來的預設值；蒙皮做好之後，服飾本來
# 就會跟著骨架動到任何姿勢（見 pose_garment 的說明），角度不再需要配合
# 特定服飾寫死，所以這裡可以另外定義三種常見的人體展示姿勢：
#   T-Pose  ：SMPL-X 原始姿勢，雙臂完全打平，角度 = 0
#   A-Pose  ：雙臂略往下垂，跟身體夾角約 40 度，動畫/建模業界常見的預設檢視姿勢
#   自然站立：雙臂自然垂放貼近身體兩側 + 手肘微彎
#
# 自然站立原本直接沿用 MEASURE_ARM_ANGLE（1.45 rad，量身體尺寸用的姿勢），
# 但那個角度是為了「量測時手臂要完全讓開軀幹」校出來的，拿來當展示姿勢
# 手臂會整條貼死在身體側面，視覺上很僵硬（截圖看起來像立正站好，不是放
# 鬆站立）。實測用蒙皮權重分出手臂／軀幹頂點量兩者最小距離，純降低角度
# 沒什麼用（肩關節單軸旋轉，手臂末端本來就會隨角度增加而貼近身體中線，
# 這是幾何上的必然，不是角度沒調好）；但同時把角度從 1.45 降到 1.25、
# 手肘加一點自然彎曲（0.15 rad），手臂中段到軀幹的中位數間距從 2.5cm
# 提升到 5.5cm，肉眼看起來明顯不再是整條貼死的僵硬感——真人放鬆站立時
# 手肘本來也不會完全打直，這個微彎同時讓姿勢更自然。
STANDING_SHOULDER_ANGLE = 1.25
STANDING_ELBOW_BEND = 0.15

POSE_PRESETS = {
    'tpose': (0.0, 0.0),
    'apose': (0.7, 0.0),
    'standing': (STANDING_SHOULDER_ANGLE, STANDING_ELBOW_BEND),
}


def _resolve_body_pose(pose):
    import torch
    if pose in POSE_PRESETS:  # 'tpose' / 'apose' / 'standing'
        shoulder_angle, elbow_bend = POSE_PRESETS[pose]
        return _natural_body_pose(shoulder_angle, elbow_bend)
    if pose == 'natural':  # 舊的服飾對齊預設姿勢，保留給直接呼叫 API 的地方相容用
        return _natural_body_pose()
    return torch.as_tensor(pose, dtype=torch.float32).reshape(1, -1)


_SMPLX_MODEL_CACHE = {}


def get_smplx_model(gender='neutral', model_dir=SMPLX_MODEL_DIR):
    """
    smplx.create() 每次呼叫都要重新從硬碟讀 108MB 的 .npz（實測約 630ms），
    對一個要即時回應滑桿拖曳的網頁來說太慢。這裡把載好的模型物件依
    (gender, model_dir) 快取起來，之後只需要重跑 forward pass（實測約 5ms）。
    """
    key = (gender, model_dir)
    if key not in _SMPLX_MODEL_CACHE:
        has_weights = os.path.isdir(model_dir) and any(f.endswith(('.npz', '.pkl')) for f in os.listdir(model_dir))
        if not has_weights:
            raise FileNotFoundError(
                f"找不到 SMPL-X 權重檔（{model_dir} 底下沒有 .npz/.pkl）。\n"
                "請先到 https://smpl-x.is.tue.mpg.de/ 註冊帳號、同意授權條款下載模型，"
                "把檔案放進這個資料夾後再呼叫 load_smplx_body()。\n"
                "在那之前可以先用 build_proxy_mannequin() 測試對齊邏輯。"
            )
        import smplx
        # smplx.create 預期傳入的是「models 根目錄」，底下要有 smplx/SMPLX_xxx.npz 這個結構，
        # model_dir 本身已經是 assets/smplx/，所以要往上一層傳給它。
        _SMPLX_MODEL_CACHE[key] = smplx.create(os.path.dirname(model_dir), model_type='smplx',
                                                gender=gender, use_pca=False)
    return _SMPLX_MODEL_CACHE[key]


def load_smplx_body(betas=None, gender='neutral', model_dir=SMPLX_MODEL_DIR,
                     pose='natural'):
    """
    載入真正的 SMPL-X 模型，輸出頂點與面（betas=None 時為平均體型）。

    pose:
        'tpose'    - SMPL-X 原始 T-pose，雙臂完全打平
        'apose'    - A-Pose，雙臂略往下垂約 40 度
        'standing' - 自然站立，雙臂自然垂放貼近身體兩側
        'natural'  - 舊的服飾對齊預設姿勢（0.5 rad），保留給既有呼叫端相容用
        或直接傳自訂的 (1, 63) body_pose tensor

    需要事先到 https://smpl-x.is.tue.mpg.de/ 註冊、同意授權後下載模型權重，
    放進 model_dir（預設 assets/smplx/），檔名需符合 smplx 套件慣例，例如：
        assets/smplx/SMPLX_NEUTRAL.npz
    在那之前呼叫這個函式會丟出 FileNotFoundError 並附上下載說明。
    """
    import torch

    model = get_smplx_model(gender=gender, model_dir=model_dir)  # 找不到權重檔會在這裡丟 FileNotFoundError
    if betas is None:
        betas = torch.zeros((1, model.num_betas))
    else:
        betas = torch.tensor(betas, dtype=torch.float32).reshape(1, -1)

    body_pose = _resolve_body_pose(pose)

    output = model(betas=betas, body_pose=body_pose, return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze(0)
    faces = model.faces
    return vertices, faces


def align_body_to_garment(body_vertices, garment_vertices):
    """
    對齊：把人體網格擺到服飾網格的座標系裡對的高度。

    不能直接比兩者的 bounding box——人體有頭有腳，服飾只是軀幹一小段，
    整體對齊一定對不準。也刻意不做「縮放對齊」：build_proxy_mannequin() 已經
    是用使用者輸入的身高／胸圍／腰圍／臀圍（公分）直接換算成公尺，跟 MGN
    服飾網格一樣是真實世界公尺尺度，兩邊單位系統本來就該一致——如果再用
    一個「垂直落點跨距」去反推縮放比例，量到的其實是我自己 HEIGHT_RATIOS
    假設的準確度誤差，反而會把身體整個放大/縮小到不合理的比例。
    所以這裡只做垂直位移：讓人體的「上衣下擺高度」對齊服飾下擺的 Y 座標，
    X/Z 保持原尺寸不動，兩者是否合身（穿模與否）留給後面的穿模檢查去驗證。
    """
    body_y = body_vertices[:, 1]
    body_height = body_y.max() - body_y.min()
    body_hem_y = body_y.min() + HEIGHT_RATIOS['shirt_hem'] * body_height

    garment_hem_y = garment_vertices[:, 1].min()
    offset_y = garment_hem_y - body_hem_y

    aligned = body_vertices.copy()
    aligned[:, 1] += offset_y

    return aligned, 1.0  # 回傳值第二項保留縮放比例欄位（目前恆為 1.0），供未來需要時擴充
