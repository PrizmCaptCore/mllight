# Mesh Optimization (pre-deployment)

학습/추론 파이프라인과 함께 자산을 준비할 때, 메시를 간소화해 런타임 퍼포먼스를 개선할 수 있습니다. 여기서는 Open3D 기반 quadric decimation 스크립트를 제공합니다.

## 준비
- Python
- `open3d` 패키지:
  ```bash
  pip install open3d
  ```

## 사용법
- 삼각형 수 지정:
  ```bash
  python3 -m model_opt.mesh_simplify --input input.obj --output output.obj --target_faces 5000
  # 또는 설치 후 스크립트
  mesh-simplify --input input.obj --output output.obj --target_faces 5000
  ```
- 비율 지정(예: 절반으로):
  ```bash
  python3 -m model_opt.mesh_simplify --input input.glb --output output.glb --ratio 0.5
  mesh-simplify --input input.glb --output output.glb --ratio 0.5
  ```

지원 포맷: obj/ply/glb/gltf/stl 등 Open3D가 읽을 수 있는 형식. 출력은 동일 형식으로 저장됩니다.

## 참고
- 간소화 후 법선이 다시 계산됩니다. 필요시 DCC 툴에서 재확인/리베이크를 권장합니다.
- 더 고급(데이터 기반) 품질 유지/디테일 보존이 필요하면:
  - 텍스처/노멀 맵을 AI 업스케일 후 다시 베이크
  - ML 기반 리토폴로지/LOD 생성 도구와 조합
  - 품질 스코어(곡률/실루엣 유지 등)를 기준으로 자동 평가 추가

필요에 따라 배치 처리 스크립트를 추가하거나, 품질 평가 지표를 넣어 확장할 수 있습니다.
