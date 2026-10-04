"""Create a minimal valid GLB file for stub adapters."""

import json
import struct
from pathlib import Path


def _pad4(data: bytes, pad_byte: bytes) -> bytes:
    pad = (4 - len(data) % 4) % 4
    return data + pad_byte * pad


def create_minimal_glb(path: Path):
    """Write a tiny but visible box GLB so the frontend can orbit a real mesh."""
    path.parent.mkdir(parents=True, exist_ok=True)

    faces = (
        ((-0.5, 0.0, 0.5), (0.5, 0.0, 0.5), (0.5, 1.0, 0.5), (-0.5, 1.0, 0.5), (0.0, 0.0, 1.0)),
        ((0.5, 0.0, -0.5), (-0.5, 0.0, -0.5), (-0.5, 1.0, -0.5), (0.5, 1.0, -0.5), (0.0, 0.0, -1.0)),
        ((0.5, 0.0, 0.5), (0.5, 0.0, -0.5), (0.5, 1.0, -0.5), (0.5, 1.0, 0.5), (1.0, 0.0, 0.0)),
        ((-0.5, 0.0, -0.5), (-0.5, 0.0, 0.5), (-0.5, 1.0, 0.5), (-0.5, 1.0, -0.5), (-1.0, 0.0, 0.0)),
        ((-0.5, 1.0, 0.5), (0.5, 1.0, 0.5), (0.5, 1.0, -0.5), (-0.5, 1.0, -0.5), (0.0, 1.0, 0.0)),
        ((-0.5, 0.0, -0.5), (0.5, 0.0, -0.5), (0.5, 0.0, 0.5), (-0.5, 0.0, 0.5), (0.0, -1.0, 0.0)),
    )

    positions = []
    normals = []
    indices = []
    for face_index, (a, b, c, d, normal) in enumerate(faces):
        base = face_index * 4
        for vertex in (a, b, c, d):
            positions.extend(vertex)
            normals.extend(normal)
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    pos_bytes = struct.pack("<" + "f" * len(positions), *positions)
    nrm_bytes = struct.pack("<" + "f" * len(normals), *normals)
    idx_bytes = struct.pack("<" + "H" * len(indices), *indices)
    pos_padded = _pad4(pos_bytes, b"\x00")
    nrm_padded = _pad4(nrm_bytes, b"\x00")
    idx_padded = _pad4(idx_bytes, b"\x00")
    bin_blob = pos_padded + nrm_padded + idx_padded

    gltf = {
        "asset": {"version": "2.0", "generator": "3Dclothes stub"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "StubBody"}],
        "meshes": [
            {
                "name": "StubBox",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.78, 0.64, 0.42, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.55,
                }
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 24,
                "type": "VEC3",
                "min": [-0.5, 0.0, -0.5],
                "max": [0.5, 1.0, 0.5],
            },
            {"bufferView": 1, "componentType": 5126, "count": 24, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes)},
            {"buffer": 0, "byteOffset": len(pos_padded), "byteLength": len(nrm_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(pos_padded) + len(nrm_padded),
                "byteLength": len(idx_bytes),
            },
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
    }

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_padded = _pad4(bin_blob, b"\x00")
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_padded)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(bin_padded), 0x004E4942)
        + bin_padded
    )
