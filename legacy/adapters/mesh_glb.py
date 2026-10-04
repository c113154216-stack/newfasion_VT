"""
numpy 網格 <-> GLB 轉換工具，給紀泓宇的 M2~M4（body_generator / garment_processor /
physics_simulator）共用。獨立成新檔案，不動 glb_utils.py（stub 時期用來產生空白
佔位 GLB 的檔案，其他 adapter 目前沒有依賴它，先保留不動）。

GLB 本身沒有欄位可以存 SMPL-X 的 betas（形狀係數），但 M4 要把服飾正確蒙皮貼合
到人體上，需要知道人體是用哪組 betas 生成的——所以 body_generator 額外寫一個
同名的 .json 「側車檔」存 betas/gender，M4 讀 body_glb 時一併讀回來。
"""
import json
from pathlib import Path

import numpy as np
import trimesh


def export_glb(vertices, faces, path):
    """把 (N,3) 頂點 + (M,3) 三角形面輸出成 .glb。"""
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,  # 不要讓 trimesh 自動合併/清理頂點，保留原始拓樸索引
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def load_glb(path):
    """讀回 .glb，回傳 (vertices, faces)，都是 numpy array。"""
    mesh = trimesh.load(str(path), process=False, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def export_glb_textured(vertices, faces, uv, texture_image, path):
    """
    export_glb() 的加強版：多帶 UV 座標 + 貼圖圖片，輸出的 .glb 在瀏覽器裡
    會顯示真正的花色，不是預設灰色。

    這是新增的獨立函式，沒有動到 export_glb() 本身——現有呼叫 export_glb() 的地方
    (garment_processor.py 等) 完全不受影響，行為不變。要換成有貼圖的版本，
    呼叫端自己改叫這支新函式即可。

    vertices/faces: 跟 export_glb() 一樣
    uv: (N, 2) ndarray，每個頂點對應的 UV 座標，範圍 [0, 1]
        （例如用 _uv_prototype_test.py 裡驗證過的 compute_uv() 算出來的那組）
    texture_image: PIL.Image，要貼上去的圖片（商品照片或測試用棋盤格都可以）
    """
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_image,
        metallicFactor=0.0,   # 避免預設全金屬，貼圖顏色才顯示得出來
        roughnessFactor=1.0,
    )
    visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv, dtype=np.float64), material=material)

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
    mesh.fix_normals()  # 修正面朝向；對封閉網格（例如 tshirt_base.obj）已驗證有效

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def load_glb_with_material(path):
    """
    load_glb() 的加強版：連 UV 座標跟貼圖圖片一起讀回來，不像 load_glb() 只回傳
    幾何資料、把材質丟掉。

    這是要接進 physics_simulator.py 才有意義的函式——目前 physics_simulator.run()
    是呼叫 load_glb() 把服裝 .glb 讀回來重組最終場景，材質在這一步會被完全丟掉；
    要保留貼圖，必須改呼叫這支函式，並在重組最終 Scene 時把 uv/texture_image
    一併帶進新的 Trimesh。這支函式先寫好，實際串接留給要改 physics_simulator.py
    的人決定怎麼接。

    回傳: (vertices, faces, uv, texture_image)
        uv 或 texture_image 若原檔案沒有貼圖資訊，會是 None。
    """
    mesh = trimesh.load(str(path), process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    uv = None
    texture_image = None
    visual = mesh.visual
    if hasattr(visual, "uv") and visual.uv is not None:
        uv = np.asarray(visual.uv, dtype=np.float64)
    material = getattr(visual, "material", None)
    if material is not None:
        texture_image = getattr(material, "baseColorTexture", None)

    return vertices, faces, uv, texture_image


def _sidecar_path(glb_path) -> Path:
    return Path(str(glb_path) + ".json")


def save_sidecar(glb_path, data: dict):
    """存 GLB 檔案本身裝不下的中繼資訊（例如 betas/gender）。"""
    _sidecar_path(glb_path).write_text(json.dumps(data), encoding="utf-8")


def load_sidecar(glb_path) -> dict:
    """讀回 save_sidecar() 存的中繼資訊；沒有的話回傳空 dict（呼叫端要自己處理沒有的情況）。"""
    p = _sidecar_path(glb_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
