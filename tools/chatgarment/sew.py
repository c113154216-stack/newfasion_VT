# -*- coding: utf-8 -*-
"""
把 GarmentCode 的 box mesh（版片擺在身體周圍、縫線還沒拉合）縫成一件完整的衣服。

box mesh 裡縫線兩側的頂點已經合併，只是縫線附近的三角形被拉成細長的「橋」。
GarmentCode 把這些邊在 2D 版片上的原始長度存在 *_orig_lens.pickle；其他邊沒被拉伸，
現在的長度就是原始長度。縫合就是讓每條邊回到原始長度。

這是純幾何的求解，不是物理模擬：沒有重力、摩擦、動態（那些是紀泓宇 XPBD 的工作）。
給了身體網格時，會加上「頂點不能進到身體裡」的限制，讓縫好的衣服停在身體外側，
當作物理模擬的起始狀態。

跑法（在 newfashion_VT 根目錄，先跑過 design_to_mesh.py）：
    .venv\\Scripts\\python tools\\chatgarment\\sew.py output\\chatgarment\\shirt_mean_male [身體.obj]

輸出：同一資料夾的 *_sewn.obj（UV、面與原檔相同，只換頂點位置），並印出縫合前後的網格品質
"""
import pickle
import sys
from pathlib import Path

import igl
import numpy as np


def mesh_edges(faces):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def rest_lengths(vertices, edges, orig_lens):
    rest = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    for k, (a, b) in enumerate(edges):
        length = orig_lens.get((a, b), orig_lens.get((b, a)))
        if length is not None:
            rest[k] = length
    return rest


def push_out(x, body_v, body_f, body_n, margin):
    """把在身體裡面、或離身體表面不到 margin 的頂點，推到表面外 margin 的位置。"""
    inside = igl.fast_winding_number_for_meshes(body_v, body_f, x) > 0.5
    d2, face, closest = igl.point_mesh_squared_distance(x, body_v, body_f)
    bad = inside | (np.sqrt(d2) < margin)
    x[bad] = closest[bad] + body_n[face[bad]] * margin
    return x


def sew(vertices, edges, rest, body=None, margin=0.5, body_weight=10.0, max_step=1.0,
        iterations=400, anchor=1e-4, stretch_limit=None, stretch_weight=20.0, gravity=0.0,
        stretch_mask=None, bend=0.0):
    """局部－全域交替求解，最小化
        Σ w_e(|xi − xj| − 原始長度)²  ＋  body_weight · Σ|xi − 推到身體外的位置|²  ＋  gravity · Σ yi
    局部：每條邊依目前方向算出長度剛好等於原始長度的目標向量；每個頂點算出「不在身體裡」的目標位置
          （本來就在外面的頂點，目標就是自己）。
    全域：解一次稀疏線性方程組（矩陣固定，只分解一次）。
    不能進身體是軟限制，跟邊長一起求平衡；每輪硬推頂點會和邊長互相拉扯，把網格扭壞。
    anchor 是很小的權重，把整件衣服拉向原位置，只為了固定整體平移、讓方程組可解。

    stretch_limit：邊被拉長超過這個比例（例如 0.05 ＝ 5%）時，該邊的權重 w_e 改成 stretch_weight。
        推身體的權重（10）比一般邊（1）大，布料卡在肩膀、脖子這類斜面時會選擇被拉長；
        把拉太長的邊變硬，布料就改成滑開。None 表示不限制（所有邊權重 1）。
    stretch_mask：長度 m 的布林陣列，只有 True 的邊受 stretch_limit 限制；None 表示全部的邊。
    bend：彎曲剛度（拉普拉斯平滑），懲罰每個頂點偏離鄰居平均位置的量 bend · Σ|xi − 鄰居平均|²。
        沒有它時，布被壓縮會擠成細碎的小皺（像揉皺的紙）；加上後多出來的布會形成平順的大褶。
        太大會把布往內拉平、抹掉該有的立體感。0 表示不加。
    gravity：每個頂點往下的固定力（單位同邊長彈簧，力＝邊被拉長幾公分 × 權重）。
        0 表示沒有重力，衣服停在 box mesh 擺放的位置附近。"""
    import scipy.sparse as sp
    from scipy.sparse.linalg import factorized

    n, m = len(vertices), len(edges)
    rows = np.repeat(np.arange(m), 2)
    cols = edges.reshape(-1)
    vals = np.tile([1.0, -1.0], m)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))  # (A x)_e = xi − xj
    # 拉普拉斯矩陣：(L x)_i ＝ xi − 鄰居平均。目標 L x ＝ 0，不隨迭代改變，直接加進矩陣
    if bend > 0:
        adj = sp.csr_matrix((np.ones(2 * m), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
        deg = np.asarray(adj.sum(axis=1)).ravel()
        Lap = sp.identity(n) - sp.diags(1.0 / np.maximum(deg, 1)) @ adj
        bend_term = bend * (Lap.T @ Lap)
    else:
        bend_term = sp.csr_matrix((n, n))
    AtA = (A.T @ A + anchor * sp.identity(n) + bend_term).tocsc()
    solve = factorized(AtA)
    w = np.ones(m)

    if body is not None:
        body_v, body_f = body
        body_n = igl.per_face_normals(body_v, body_f, np.array([0.0, 0.0, 1.0]))

    x = vertices.copy()
    for _ in range(iterations):
        d = A @ x
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        refactor = False
        if stretch_limit is not None:
            over = length > rest * (1 + stretch_limit)
            if stretch_mask is not None:
                over &= stretch_mask
            w_new = np.where(over, stretch_weight, 1.0)
            if not np.array_equal(w_new, w):
                w = w_new
                AtA = (A.T @ sp.diags(w) @ A + anchor * sp.identity(n) + bend_term).tocsc()
                refactor = True
        d *= (rest / length)[:, None]
        b = A.T @ (w[:, None] * d) + anchor * vertices
        b[:, 1] -= gravity
        if body is not None:
            # 只有碰到身體（在裡面或貼太近）的頂點才加這一項；沒碰到的不加，
            # 否則「目標＝自己」會變成阻力，拖慢收斂。碰到的頂點每輪不同，所以矩陣要重新分解。
            target = push_out(x.copy(), body_v, body_f, body_n, margin)
            hit = np.any(target != x, axis=1).astype(float) * body_weight
            b += hit[:, None] * target
            solve = factorized((AtA + sp.diags(hit)).tocsc())
        elif refactor:
            solve = factorized(AtA)
        x_new = np.column_stack([solve(b[:, k]) for k in range(3)])
        if body is None:
            x = x_new
        else:
            # 每輪每個頂點最多移動 max_step cm，避免一步跳過整個身體（穿隧）而沒被碰撞擋下
            step = x_new - x
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            x = x + step * np.minimum(1.0, max_step / np.maximum(norm, 1e-9))
    return x


def write_obj_with_new_positions(src, dst, vertices):
    """沿用原 OBJ 的 vt、f、mtl，只把 v 行換成新位置。"""
    out, k = [], 0
    for line in Path(src).read_text().splitlines():
        if line.startswith("v "):
            out.append("v {:.6f} {:.6f} {:.6f}".format(*vertices[k]))
            k += 1
        else:
            out.append(line)
    assert k == len(vertices)
    Path(dst).write_text("\n".join(out) + "\n")


def quality(x, faces, edges, rest, body):
    """縫好之後交給物理模擬前要看的指標。"""
    import trimesh
    mesh = trimesh.Trimesh(x, faces, process=False)
    ratio = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1) / rest
    min_angle = np.degrees(mesh.face_angles).min(axis=1)
    report = {
        "頂點／三角面": f"{len(x)}／{len(faces)}",
        "邊長偏離原始長度超過 10%": f"{np.mean(np.abs(ratio - 1) > 0.1) * 100:.2f}%",
        "最小角 < 10° 的三角形": f"{np.mean(min_angle < 10) * 100:.2f}%",
        "最長邊": f"{mesh.edges_unique_length.max():.2f} cm",
    }
    if body is not None:
        bv, bf = body
        inside = igl.fast_winding_number_for_meshes(bv, bf, x) > 0.5
        dist = np.sqrt(igl.point_mesh_squared_distance(x, bv, bf)[0])
        report["在身體裡的頂點"] = f"{inside.mean() * 100:.2f}%"
        report["離身體距離（中位數）"] = f"{np.median(dist):.1f} cm"
    return report


def main(folder, body_obj=None):
    folder = Path(folder)
    src = next(folder.glob("*_boxmesh.obj"))
    v, f = igl.read_triangle_mesh(str(src))
    orig_lens = pickle.loads(next(folder.glob("*_orig_lens.pickle")).read_bytes())

    body = None
    if body_obj:
        bv, bf = igl.read_triangle_mesh(str(body_obj))
        body = (bv * 100, bf)  # GarmentCode 的身體網格單位是公尺，衣服是公分

    edges = mesh_edges(f)
    rest = rest_lengths(v, edges, orig_lens)
    x = sew(v, edges, rest, body=body)

    dst = src.with_name(src.name.replace("_boxmesh.obj", "_sewn.obj"))
    write_obj_with_new_positions(src, dst, x)
    print(dst)
    for tag, verts in [("縫合前", v), ("縫合後", x)]:
        print(f"[{tag}]")
        for k, val in quality(verts, f, edges, rest, body).items():
            print(f"  {k}：{val}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
