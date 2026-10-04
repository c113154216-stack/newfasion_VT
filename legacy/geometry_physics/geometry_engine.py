# -*- coding: utf-8 -*-
"""
3D 服飾幾何演算法核心模組。

對應計畫書模組二/三（參數化體型生成、3D 服飾幾何校正）中「頂點拉伸」的
後端實作。公版短袖上衣由兩部分組成：

    1. 軀幹 (torso)：用「輪廓內插 + 旋轉延伸 (lathe)」建立橢圓截面的中空管狀
       網格，寬度沿高度用 smoothstep 內插出腰／胸／肩的曲線，並在領口處
       依前後中心挖出 U 型領。
    2. 袖子 (sleeve)：各自獨立建模成一根從肩點朝外、略微下垂的中空圓管，
       用 Rodrigues 旋轉矩陣把局部座標對齊到肩袖方向後接上軀幹。

尺寸調整同樣不是逐分量純量相乘，而是：
    1. 每個部位（肩／胸／腰／基礎）各自對應一個 4x4 齊次縮放矩陣 M_i
    2. 每個頂點在「建網格當下」就依所屬部位算好權重 w_i（袖子頂點固定
       100% 歸屬肩部），且 sum(w_i) = 1
    3. 最終頂點 = Σ w_i * (M_i @ v_homogeneous)

這是 Linear Blend Skinning (LBS) 的簡化版本，與 SMPL-X 以權重混合多個
骨骼變換矩陣的概念一致，方便未來替換成真正的骨骼/形狀混合權重。
"""

import numpy as np

# ---- 軀幹輪廓控制點：(高度 t∈[0,1]，x 方向半寬) ----
# t=0 為下擺，t=1 為肩線頂端；領口的凹陷另外用 _apply_neckline 處理。
TORSO_PROFILE = [
    (0.00, 0.62),  # 下擺
    (0.40, 0.62),  # 腰部（維持筆直，接近下擺寬度，呈直筒版型）
    (0.62, 0.72),  # 胸部（軀幹最寬處，僅略寬於下擺）
    (0.80, 0.70),  # 肩線／腋下（袖子由此接出）
    (1.00, 0.70),  # 肩線頂端（領口由 _apply_neckline 挖出）
]
DEPTH_RATIO = 0.42          # 前後厚度 = x 方向半寬 * DEPTH_RATIO（讓截面呈扁橢圓，像布料）
TORSO_HEIGHT_SEGMENTS = 48
TORSO_RADIAL_SEGMENTS = 48

SLEEVE_RADIAL_SEGMENTS = 16
SLEEVE_LENGTH_SEGMENTS = 6
SLEEVE_ANGLE_DOWN_DEG = 22   # 袖子相對水平線向下傾斜角度
SLEEVE_LENGTH = 0.62
SLEEVE_ATTACH_T = 0.80       # 對應 TORSO_PROFILE 的肩線位置
SLEEVE_ATTACH_INSET = 0.92   # 讓袖根略縮進軀幹表面內側，避免與軀幹之間出現縫隙


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _profile_half_width(t):
    """依控制點分段用 smoothstep 內插出軀幹 x 方向半寬，避免線性內插的明顯折角。"""
    t = np.asarray(t, dtype=np.float64)
    keys = np.array([p[0] for p in TORSO_PROFILE])
    widths = np.array([p[1] for p in TORSO_PROFILE])

    result = np.full(t.shape, widths[-1], dtype=np.float64)
    result[t <= keys[0]] = widths[0]
    for i in range(len(keys) - 1):
        mask = (t > keys[i]) & (t <= keys[i + 1])
        local_t = _smoothstep(keys[i], keys[i + 1], t[mask])
        result[mask] = widths[i] + (widths[i + 1] - widths[i]) * local_t
    return result


def _region_weights_torso(t):
    """依高度計算腰／胸／肩三個部位的混合權重，三者相加恆為 1。"""
    waist_w = 1.0 - _smoothstep(0.30, 0.45, t)
    shoulder_w = _smoothstep(0.68, 0.85, t)
    chest_w = np.clip(1.0 - waist_w - shoulder_w, 0.0, 1.0)
    base_w = np.zeros_like(t)
    return base_w, shoulder_w, chest_w, waist_w


def _apply_neckline(X, Y, Z, T, THETA):
    """在肩線頂端、前後中心 (theta≈0 或 π，即 |sin(theta)| 小) 處把 Y 往下拉，挖出 U 型領口。"""
    neck_zone = _smoothstep(0.90, 1.0, T)
    center_factor = np.clip(1.0 - np.abs(np.sin(THETA)) / 0.32, 0.0, 1.0)
    dip = neck_zone * center_factor * 0.18
    return Y - dip


def _build_torso():
    hs, rs = TORSO_HEIGHT_SEGMENTS, TORSO_RADIAL_SEGMENTS
    t = np.arange(hs + 1) / hs
    theta = (np.arange(rs + 1) / rs) * 2.0 * np.pi

    T, THETA = np.meshgrid(t, theta, indexing='ij')  # (hs+1, rs+1)

    half_w = _profile_half_width(T)
    half_d = half_w * DEPTH_RATIO

    X = half_w * np.sin(THETA)
    Z = half_d * np.cos(THETA)
    Y = -1.0 + 2.0 * T
    Y = _apply_neckline(X, Y, Z, T, THETA)

    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    base_w, shoulder_w, chest_w, waist_w = _region_weights_torso(T.ravel())
    weights = np.stack([base_w, shoulder_w, chest_w, waist_w], axis=1)

    faces = _grid_faces(hs, rs)
    return vertices, weights, faces


def _rotation_from_to(a, b):
    """回傳把單位向量 a 旋轉對齊到單位向量 b 的 3x3 旋轉矩陣 (Rodrigues' rotation formula)。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


def _build_sleeve(sign):
    """建立一根中空圓管袖子。sign=+1 為右袖，sign=-1 為左袖。"""
    rs, ls = SLEEVE_RADIAL_SEGMENTS, SLEEVE_LENGTH_SEGMENTS

    shoulder_half_width = float(_profile_half_width(np.array([SLEEVE_ATTACH_T]))[0])
    shoulder_y = -1.0 + 2.0 * SLEEVE_ATTACH_T
    attach_point = np.array([sign * shoulder_half_width * SLEEVE_ATTACH_INSET, shoulder_y, 0.0])

    angle_down = np.radians(SLEEVE_ANGLE_DOWN_DEG)
    direction = np.array([sign * np.cos(angle_down), -np.sin(angle_down), 0.0])

    top_radius = shoulder_half_width * 0.42
    cuff_radius = top_radius * 0.72

    u = np.arange(ls + 1) / ls          # 0 (肩) ~ 1 (袖口)
    theta = (np.arange(rs + 1) / rs) * 2.0 * np.pi
    U, THETA = np.meshgrid(u, theta, indexing='ij')

    radius = top_radius + (cuff_radius - top_radius) * U
    local_x = radius * np.sin(THETA)
    local_z = radius * np.cos(THETA)
    local_y = U * SLEEVE_LENGTH

    local = np.stack([local_x.ravel(), local_y.ravel(), local_z.ravel()], axis=1)

    R = _rotation_from_to(np.array([0.0, 1.0, 0.0]), direction)
    world = local @ R.T + attach_point

    weights = np.zeros((world.shape[0], 4))
    weights[:, 1] = 1.0  # 袖子完全跟隨「肩寬」矩陣縮放

    faces = _grid_faces(ls, rs)
    return world, weights, faces


def _grid_faces(row_segments, col_segments):
    """把 (row_segments+1) x (col_segments+1) 的頂點網格串成三角形面。"""
    cols = col_segments + 1
    ai, aj = np.meshgrid(np.arange(row_segments), np.arange(col_segments), indexing='ij')
    a = (ai * cols + aj).ravel()
    b = a + 1
    c = a + cols
    d = c + 1
    tri1 = np.stack([a, c, d], axis=1)
    tri2 = np.stack([a, d, b], axis=1)
    return np.vstack([tri1, tri2])


def load_obj_vf(path):
    """
    最小可用的 OBJ 讀取器（只取頂點座標與三角形面索引，1-based -> 0-based）。
    用來讀取外部資料集（如 MGN Multi-Garment）提供的真實服飾掃描網格。
    """
    verts = []
    faces = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def build_real_shirt_template(obj_path):
    """
    載入外部真實服飾掃描網格（例如 MGN Multi-Garment 資料集的 TShirtNoCoat.obj），
    自動把頂點分成「袖子」與「軀幹」兩群並分別計算 base/shoulder/chest/waist 混合權重，
    回傳格式與 build_base_shirt() 一致，可直接供 deform_vertices() 使用。

    分類邏輯：軀幹在下半身 (t<0.5) 的橫向延伸 (|x|) 有一個穩定的上限，
    袖子頂點會遠遠超出這個上限，因此用「下半身最大橫向延伸 * 安全係數」
    當作門檻，超過門檻的頂點視為袖子，其餘視為軀幹。這比單純按高度切一刀
    準確得多——避免調整「肩寬」時連帶把領口/衣領也一併撐開的怪異變形。
    """
    vertices, faces = load_obj_vf(obj_path)
    y = vertices[:, 1]
    x = vertices[:, 0]
    t = (y - y.min()) / (y.max() - y.min())

    # 下半身必定不含袖子，用它的橫向延伸上限估計「純軀幹」的寬度基準
    torso_reference_width = np.abs(x[t < 0.5]).max()
    sleeve_threshold = torso_reference_width * 1.15
    is_sleeve = np.abs(x) > sleeve_threshold

    weights = np.zeros((vertices.shape[0], 4))

    # 袖子頂點：完全跟隨「肩寬」矩陣縮放
    weights[is_sleeve, 1] = 1.0

    # 軀幹頂點：腰部（下擺附近）、胸部（中段）、肩部（僅在腋下~肩線這個窄帶，
    # 過了衣領範圍要淡出，避免領口跟著「肩寬」一起被撐大）
    torso_t = t[~is_sleeve]
    waist_w = 1.0 - _smoothstep(0.30, 0.45, torso_t)
    shoulder_w = _smoothstep(0.60, 0.75, torso_t) * (1.0 - _smoothstep(0.90, 1.0, torso_t))
    chest_w = np.clip(1.0 - waist_w - shoulder_w, 0.0, 1.0)

    weights[~is_sleeve, 1] = shoulder_w
    weights[~is_sleeve, 2] = chest_w
    weights[~is_sleeve, 3] = waist_w

    return vertices, weights, faces


def build_base_shirt():
    """
    組出完整的公版短袖上衣（軀幹 + 左右袖），回傳：
        vertices: (N, 3) 頂點座標
        weights:  (N, 4) 每個頂點對 base/shoulder/chest/waist 的混合權重
        faces:    (F, 3) 三角形頂點索引
    """
    torso_v, torso_w, torso_f = _build_torso()
    right_v, right_w, right_f = _build_sleeve(+1)
    left_v, left_w, left_f = _build_sleeve(-1)

    offset_right = torso_v.shape[0]
    offset_left = offset_right + right_v.shape[0]

    vertices = np.vstack([torso_v, right_v, left_v])
    weights = np.vstack([torso_w, right_w, left_w])
    faces = np.vstack([torso_f, right_f + offset_right, left_f + offset_left])

    return vertices, weights, faces


def _scale_matrix(sx, sy, sz):
    """建立 4x4 齊次座標縮放矩陣。"""
    M = np.eye(4)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def deform_vertices(vertices, weights, params):
    """
    對輸入頂點套用尺寸參數，回傳變形後的頂點。

    vertices: (N, 3) ndarray，公版服飾頂點
    weights:  (N, 4) ndarray，對應每個頂點的 base/shoulder/chest/waist 混合權重
    params:   dict，包含 length / shoulder / chest / waist，數值以 100 為原尺寸
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if vertices.size == 0:
        return vertices

    length_scale = float(params.get('length', 100)) / 100.0
    shoulder_scale = float(params.get('shoulder', 100)) / 100.0
    chest_scale = float(params.get('chest', 100)) / 100.0
    waist_scale = float(params.get('waist', 100)) / 100.0

    M_base = _scale_matrix(1.0, length_scale, 1.0)
    M_shoulder = _scale_matrix(shoulder_scale, length_scale, 1.0 + (shoulder_scale - 1.0) * 0.6)
    M_chest = _scale_matrix(chest_scale, length_scale, chest_scale)
    M_waist = _scale_matrix(waist_scale, length_scale, waist_scale)

    homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])  # (N, 4)

    v_base = homogeneous @ M_base.T
    v_shoulder = homogeneous @ M_shoulder.T
    v_chest = homogeneous @ M_chest.T
    v_waist = homogeneous @ M_waist.T

    w_base, w_shoulder, w_chest, w_waist = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]

    blended = (w_base[:, None] * v_base +
               w_shoulder[:, None] * v_shoulder +
               w_chest[:, None] * v_chest +
               w_waist[:, None] * v_waist)

    return blended[:, :3]


def export_obj(vertices, faces, path):
    """把頂點+三角形面輸出成 .obj，方便用一般 3D 檢視器（如 Blender）目視檢查結果。"""
    with open(path, 'w', encoding='utf-8') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            # OBJ 索引從 1 開始
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
