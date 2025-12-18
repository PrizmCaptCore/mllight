# Godot On-Device LLM (GGUF + llama.cpp via GDExtension)

이 폴더는 Godot에서 GGUF 모델을 직접 로드해 오프라인 추론을 하는 스캐폴드입니다. 기본 아이디어는 Unity 통합과 같으며, LoRA를 병합해 GGUF로 변환한 뒤 llama.cpp를 GDExtension으로 래핑해 GDScript에서 호출합니다.

## 0) 전제
- LoRA 병합 → GGUF 변환 완료 (예: `qwen15b.gguf`, `tokenizer.json`)
- Godot 4.x
- 플랫폼별로 llama.cpp를 빌드할 수 있는 toolchain

## 1) 디렉터리 구조 예시
```
inference/godot/
  README.md
  addons/
    llama_gd/
      godot.llama_gd.gdextension   # GDExtension 설정
      src/
        register_types.cpp         # 래퍼 등록
        llama_wrapper.cpp          # 간단한 C API 호출 래퍼
        CMakeLists.txt             # 빌드 스크립트 (플랫폼별)
      bin/
        libllama_gd.so / .dll / .dylib
  models/
    qwen15b.gguf
    tokenizer.json
  demo/
    LlamaDemo.tscn
    LlamaDemo.gd                  # GDScript 예시
```

## 2) GDExtension 최소 코드 스케치
- `register_types.cpp`
```cpp
#include "register_types.h"
#include "llama_wrapper.h"

void initialize_llama_gd(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    ClassDB::register_class<LlamaWrapper>();
}

void uninitialize_llama_gd(ModuleInitializationLevel p_level) { }
```

- `llama_wrapper.h/.cpp` (llama.cpp C API를 간단히 감싼 클래스)
```cpp
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
extern "C" {
#include "llama.h"
}

using namespace godot;

class LlamaWrapper : public RefCounted {
    GDCLASS(LlamaWrapper, RefCounted);
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

protected:
    static void _bind_methods();

public:
    LlamaWrapper() = default;
    ~LlamaWrapper() {
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
    }

    bool load_model(String path, int ctx_size = 2048, int threads = 8);
    String generate(String prompt, int max_tokens = 120, float temperature = 0.4, float repeat_penalty = 1.1);
};
```
> 실제 구현에서는 llama.cpp의 샘플 코드를 참고해 토크나이즈→디코드→샘플링 루프를 작성합니다. 프로젝트에 `llama.cpp`를 서브모듈로 포함하고, `CMakeLists.txt`에서 정적으로 링크하거나 shared로 빌드하세요.

## 3) GDScript 사용 예시 (`demo/LlamaDemo.gd`)
```gdscript
extends Node

@onready var llama = LlamaWrapper.new()

func _ready():
    var model_path = ProjectSettings.globalize_path("res://models/qwen15b.gguf")
    var ok = llama.load_model(model_path, 2048, 8)
    if not ok:
        push_error("Failed to load model")
        return
    var prompt = "Player: 안녕\nNPC:"
    var reply = llama.generate(prompt, 120, 0.4, 1.1)
    print("NPC:", reply)
```

## 4) 빌드 힌트
- Godot C++ 템플릿(GDExtension) 초기화: `scons platform=windows|linux|macos` 또는 CMake 기반 빌드 사용.
- llama.cpp 빌드 옵션:
  - CPU 전용: 기본
  - GPU: `LLAMA_CUBLAS`, `LLAMA_METAL`, `LLAMA_CLBLAST` 등 플랫폼별 옵션을 켜서 빌드 후 `llama.cpp` 라이브러리를 링크.
- 산출물(`libllama_gd.*`)을 `addons/llama_gd/bin/`에 두고, `godot.llama_gd.gdextension`에서 각 플랫폼 바이너리를 선언.

## 5) RAG 붙이기 (선택)
- 임베딩 모델 GGUF(e5-small 등)를 별도로 로드해 코사인 검색 구현, 또는 사전 계산한 벡터/색인 파일을 번들링하여 GDScript에서 단순 검색만 수행.

## 6) 간단 대안: 서버형 호출
- 온디바이스가 부담되면 FastAPI 등으로 `/chat` REST를 띄우고 Godot `HTTPRequest`로 호출하는 방식이 가장 간단합니다.

필요에 따라 C++ 래퍼 코드/빌드 스크립트를 더 구체화할 수 있으니, 목표 플랫폼(Win/Linux/macOS/모바일)과 GPU 사용 여부를 알려주시면 맞춰 드리겠습니다.
