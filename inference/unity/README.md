# Unity Integration (Local LLM with GGUF)

이 디렉터리는 학습한 LoRA를 병합한 뒤 GGUF로 변환하여 Unity에서 로컬 추론을 돌리기 위한 스캐폴드입니다. 기본 흐름은 아래와 같습니다.

## 1) LoRA 병합 (Python, 한 번만 실행)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "Qwen/Qwen2.5-1.5B-Instruct"          # 베이스 모델
LORA = "trained_models/qwen15b_high_quality" # 학습된 LoRA 체크포인트
OUT = "export/qwen15b_merged"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, LORA)
model = model.merge_and_unload()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
```

## 2) GGUF 변환 (llama.cpp `convert-hf-to-gguf.py`)
llama.cpp 리포를 가져왔다고 가정합니다.
```bash
python3 convert-hf-to-gguf.py --outfile qwen15b.gguf --q4_k_m export/qwen15b_merged
```
산출물:
- `qwen15b.gguf` (모델)
- `tokenizer.json` (토크나이저)

## 3) Unity 프로젝트에 배치
```
inference/unity/
  README.md
  Assets/
    Scripts/
      LlamaRunner.cs        # GGUF 로드/추론 래퍼 (LlamaSharp 사용 예시)
    StreamingAssets/
      models/
        qwen15b.gguf
        tokenizer.json
```

## 4) C# 측 추론 (LlamaSharp 예시)
`Assets/Scripts/LlamaRunner.cs` 참고. 요약:
```csharp
using LLama;
using LLama.Common;

var ctx = new LLamaContext(
    LLamaWeights.LoadFromFile("path/to/qwen15b.gguf"),
    new LLamaContextParams { ContextSize = 2048, Threads = 8 }
);
var exe = new StatelessExecutor(ctx);
var prompt = "Player: hello\nNPC:";
var reply = exe.Infer(prompt, new InferenceParams {
    MaxTokens = 120,
    Temperature = 0.4f,
    RepeatPenalty = 1.1f
});
```

## 5) 대안: 직접 llama.cpp 네이티브 플러그인 사용
- 플랫폼별로 `llama.cpp`를 shared library로 빌드(`llama.dll`/`libllama.so`/`libllama.dylib`) 후 `Plugins/`에 배치.
- C#에서 P/Invoke로 `llama_init_from_file`, `llama_context_default_params`, `llama_batch`, `llama_sample` 등을 호출.
- 더 빠른 통합을 원하면 LlamaSharp 같은 .NET 래퍼를 추천합니다.

## 6) RAG 붙이기 (선택)
- 임베딩 모델 GGUF(e5-small 등)를 별도로 로드해 C#에서 코사인 검색 구현, 또는 사전 계산된 벡터/색인을 번들링하여 단순 검색만 Unity에서 수행합니다.

## 플랫폼/성능 팁
- 1.5B 4bit GGUF는 CPU에서도 동작하지만, GPU 오프로딩 지원 빌드로 지연을 줄일 수 있습니다.
- 모바일/저사양에서는 컨텍스트 길이·스레드 수를 줄이고 더 작은 모델(예: 1B) 고려.

## 필요한 것
- GGUF 모델, tokenizer
- llama.cpp 또는 LlamaSharp(.NET) 패키지
- Unity 2021+ 권장
