"""肩點在 3D 被縫線往內拉了多少：用肩點附近後片內部的頂點（版片→3D 幾乎不變形）
算剛體對位，推回「版片上的肩角」在 3D 應該在哪，和實際肩點頂點比。"""
import sys, json, yaml
sys.path.insert(0, '.')
from verify_flat import *
for item in sys.argv[1:]:
    gid, size = item.split('/'); name = f'specb_{gid}_{size}'
    x, f = igl.read_triangle_mesh(str(B/gid/size/f'{gid}_{size}.obj'))
    bx, _ = igl.read_triangle_mesh(str(R/'chatgarment'/name/f'{name}_boxmesh.obj'))
    seg = np.array((R/'chatgarment'/name/f'{name}_sim_segmentation.txt').read_text().split())
    vl = yaml.safe_load(open(R/'chatgarment'/name/f'{name}_vertex_labels.yaml'))
    spec = json.loads((R/'chatgarment'/name/f'{name}_specification.json').read_text(encoding='utf-8'))['pattern']['panels']
    cn = corners(x, vl)
    pred = {}
    for side, c in cn.items():
        pn = f'{side}_btorso'; P = spec[pn]
        V = np.asarray(P['vertices'], float); t = np.asarray(P['translation'])
        # 版片上的肩角：後片頂點中 |x| 最大且在上方（肩線外端）
        top = V[V[:, 1] > V[:, 1].max() - 8]
        k = top[np.abs(top[:, 0]).argmax()]
        corner_box = np.array([k[0], k[1], 0.0]) + t  # rotation 都是 0
        m = np.flatnonzero(seg == pn)
        near = m[np.linalg.norm(bx[m] - corner_box, axis=1) < 7]
        A, Bp = bx[near], x[near]
        ca, cb = A.mean(0), Bp.mean(0)
        U, S, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
        Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1; Rm = (U @ Vt).T
        fit = np.linalg.norm((A - ca) @ Rm.T + cb - Bp, axis=1)
        pred[side] = (corner_box - ca) @ Rm.T + cb
        print(f'{item} {side}: 版片肩角應在 {pred[side].round(1)}，實際肩點頂點 {x[c].round(1)}，被拉了 {np.linalg.norm(pred[side]-x[c]):.1f} cm（對位誤差 {fit.mean():.2f}，用 {len(near)} 點）')
    print(f'  兩個「版片肩角應在位置」直線距離 {np.linalg.norm(pred["left"]-pred["right"]):.1f}；實際肩點直線 {np.linalg.norm(x[cn["left"]]-x[cn["right"]]):.1f}')
