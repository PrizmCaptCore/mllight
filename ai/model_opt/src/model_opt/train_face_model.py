"""
Train a simple face scoring model for remeshing.

입력: 하나 이상의 메쉬 경로(obj/ply/glb/gltf/stl)
- 각 면에 대한 기하 피처를 추출(face_features)
- 휴리스틱 스코어(현재 remesh_ml의 기본 점수)를 학습 타깃으로 사용

산출: joblib 모델 파일 (RandomForestRegressor)

주의: 이는 데이터 라벨이 없을 때를 위한 간단한 예제입니다.
실제 고품질 리토폴 데이터를 라벨로 제공하면 훨씬 나은 모델을 만들 수 있습니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .remesh_ml import face_features, load_mesh, score_faces


def collect_features(mesh_paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    feats_list = []
    scores_list = []
    for p in mesh_paths:
        mesh = load_mesh(p)
        feats = face_features(mesh)
        scores = score_faces(feats, model=None)  # heuristic scores as pseudo-labels
        feats_list.append(feats)
        scores_list.append(scores)
    X = np.vstack(feats_list)
    y = np.hstack(scores_list)
    return X, y


def train_model(mesh_paths: Sequence[Path], output: Path, n_estimators: int = 200, max_depth: int | None = None):
    X, y = collect_features(mesh_paths)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    joblib.dump(model, output)
    return model, X.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple face scoring model for remeshing.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Input mesh files (obj/ply/glb/gltf/stl)")
    parser.add_argument("--output", type=Path, required=True, help="Output joblib model path")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees for RandomForestRegressor")
    parser.add_argument("--max_depth", type=int, help="Max depth for RandomForestRegressor")
    return parser.parse_args()


def main():
    args = parse_args()
    model, n_samples = train_model(args.input, args.output, args.n_estimators, args.max_depth)
    print(f"Trained model on {n_samples} faces. Saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
