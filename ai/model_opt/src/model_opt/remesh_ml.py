"""
Heuristic/ML-assisted remeshing pipeline.

아이디어:
- 메시에서 간단한 기하 피처(면적, 에지 길이, 종횡비, 노멀 편차 등)를 추출
- (옵션) joblib으로 저장된 ML 모델을 불러와 면 단위 점수를 예측
- 점수가 낮은 면을 일부 제거/정리 후 quadric decimation으로 목표 면 수까지 간소화

주의: 이 코드는 학습용/실험용 스캐폴드입니다. 실제 고품질 리토폴에는 추가 후처리와 품질 평가가 필요합니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def face_features(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Compute simple per-face features: area, mean edge length, aspect ratio, normal variance."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    e0_len = np.linalg.norm(e0, axis=1)
    e1_len = np.linalg.norm(e1, axis=1)
    e2_len = np.linalg.norm(e2, axis=1)
    mean_edge = (e0_len + e1_len + e2_len) / 3.0

    # Heron formula for area
    s = (e0_len + e1_len + e2_len) * 0.5
    area = np.sqrt(np.maximum(s * (s - e0_len) * (s - e1_len) * (s - e2_len), 1e-12))

    # aspect ratio: longest / shortest edge
    min_edge = np.minimum(np.minimum(e0_len, e1_len), e2_len)
    max_edge = np.maximum(np.maximum(e0_len, e1_len), e2_len)
    aspect = np.divide(max_edge, min_edge + 1e-9)

    # normal variance: difference between per-face normal and vertex normals
    face_normals = np.asarray(mesh.triangle_normals)
    vert_normals = np.asarray(mesh.vertex_normals)
    vn0 = vert_normals[tris[:, 0]]
    vn1 = vert_normals[tris[:, 1]]
    vn2 = vert_normals[tris[:, 2]]
    vn_mean = (vn0 + vn1 + vn2) / 3.0
    normal_diff = np.linalg.norm(face_normals - vn_mean, axis=1)

    feats = np.stack([area, mean_edge, aspect, normal_diff], axis=1)
    return feats


def load_model(model_path: Optional[Path]):
    if model_path is None:
        return None
    try:
        import joblib
    except ImportError as e:
        raise ImportError("joblib is required to load the ML model. Install with extras 'ml'.") from e
    return joblib.load(model_path)


def score_faces(feats: np.ndarray, model) -> np.ndarray:
    if model is None:
        # Heuristic: 낮은 면적, 과도한 종횡비, 노멀 차이가 큰 면에 패널티
        area = feats[:, 0]
        aspect = feats[:, 2]
        normal_diff = feats[:, 3]
        # normalize
        area_n = area / (area.mean() + 1e-6)
        aspect_n = aspect / (aspect.mean() + 1e-6)
        normdiff_n = normal_diff / (normal_diff.mean() + 1e-6)
        score = 1.0 / (0.5 * area_n + 0.3 * aspect_n + 0.2 * normdiff_n + 1e-6)
        return score
    pred = model.predict(feats)
    if pred.ndim > 1:
        pred = pred.squeeze()
    return pred


def prune_faces(mesh: o3d.geometry.TriangleMesh, keep_mask: np.ndarray) -> o3d.geometry.TriangleMesh:
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    faces_kept = tris[keep_mask]
    new_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces_kept),
    )
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_vertex_normals()
    return new_mesh


def remesh(
    input_path: Path,
    output_path: Path,
    target_faces: int,
    drop_ratio: float = 0.05,
    model_path: Optional[Path] = None,
):
    mesh = load_mesh(input_path)
    feats = face_features(mesh)
    model = load_model(model_path)
    scores = score_faces(feats, model)

    # Drop lowest-scoring faces
    keep_count = int(len(scores) * (1.0 - drop_ratio))
    keep_idx = np.argsort(scores)[-keep_count:]
    keep_mask = np.zeros(len(scores), dtype=bool)
    keep_mask[keep_idx] = True

    pruned = prune_faces(mesh, keep_mask)

    simplified = pruned.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_vertices()
    simplified.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(str(output_path), simplified)
    if not success:
        raise IOError(f"Failed to write mesh: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic/ML-assisted remeshing.")
    parser.add_argument("--input", type=Path, required=True, help="Input mesh (obj/ply/glb/gltf/stl).")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh path.")
    parser.add_argument("--target_faces", type=int, required=True, help="Target triangle count.")
    parser.add_argument("--drop_ratio", type=float, default=0.05, help="Fraction of low-score faces to drop before decimation.")
    parser.add_argument("--model", type=Path, help="Optional joblib model to score faces.")
    return parser.parse_args()


def main():
    args = parse_args()
    remesh(args.input, args.output, args.target_faces, args.drop_ratio, args.model)
    print(f"Remeshed mesh saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
