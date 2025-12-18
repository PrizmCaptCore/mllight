"""
Mesh simplification helper using Open3D's quadric decimation.

이 스크립트는 학습 파이프라인과 함께 자산을 준비할 때
폴리곤 수를 줄여 추론/런타임 퍼포먼스를 개선하는 용도로 사용할 수 있습니다.
기능은 beta 상태이며, 좀 더 개발할 예정 입니다.

Usage:
  python3 mesh_simplify.py --input input.obj --output output.obj --target_faces 5000
  python3 mesh_simplify.py --input input.glb --output output.glb --ratio 0.5
"""

import argparse
import sys
from pathlib import Path

import open3d as o3d


def simplify_mesh(
    input_path: Path,
    output_path: Path,
    target_faces: int | None = None,
    ratio: float | None = None,
) -> None:
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {input_path}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if target_faces is None and ratio is None:
        raise ValueError("Specify either --target_faces or --ratio.")

    if ratio is not None:
        if not (0 < ratio < 1):
            raise ValueError("--ratio must be between 0 and 1.")
        target_faces = int(len(mesh.triangles) * ratio)

    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_vertices()
    simplified.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(str(output_path), simplified)
    if not success:
        raise IOError(f"Failed to write mesh: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mesh simplification using Open3D.")
    parser.add_argument("--input", type=Path, required=True, help="Input mesh (obj/ply/glb/gltf/stl).")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh path.")
    parser.add_argument("--target_faces", type=int, help="Target triangle count.")
    parser.add_argument("--ratio", type=float, help="Target face ratio (0~1).")
    return parser.parse_args()


def main():
    args = parse_args()
    simplify_mesh(args.input, args.output, args.target_faces, args.ratio)
    print(f"Simplified mesh saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
