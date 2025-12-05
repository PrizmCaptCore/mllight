# mllight

> ml fine tuning 모듈 LoRA 방식

# setting up for beginner

#### 1. conda로 가상 환경을 만들어주세요. python 3.12부터는 pip install을 메인 환경에 직접할 수 없습니다.
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

미니 콘다를 설치하는 linux 쪽 코드 입니다. windows는 
> Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\Downloads\Miniconda3-latest-Windows-x86_64.exe"

자세한건 anaconda 쪽에서 확인하실 수 있습니다.

    https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer

#### 2. bash, exe 실행 하셔야 합니다.

#### 3. 패키지 설치가 필요합니다.

model_build 폴더에 들어간 뒤, (즉 pyproject.toml이 있는 디렉토리에서)
``` pip install -e . ```
를 실행해주세요. (혹은 직접 패키지를 말아 저장하셔도 됩니다. whl 형태로 나올거에요.)

# 학습 코드 실행 인터페이스

#### 1. inference, build 코드를 분리하는걸 추천합니다.

> build.py
```
from pathlib import Path
from mllight_train.tunner import LoRATrainer

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    config_path = base_dir / "config" / "test_config.yaml"

    trainer = LoRATrainer(str(config_path))
    trainer.run()
```
> inference.py
```
from pathlib import Path
import json
import re
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 대화 톤을 일관되게 잡기 위한 시스템 프롬프트
SYSTEM_PROMPT = (
    "You are Hamlet, a brooding but articulate game NPC. "
    "Answer briefly in Korean with a reflective, slightly archaic tone, "
    "and stay consistent with prior context."
)


def load_trained_model(base_model_id: str, lora_path: str):
    """
    학습된 LoRA 모델 로드

    Args:
        base_model_id: 베이스 모델 ID (예: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        lora_path: LoRA 어댑터 경로 (예: "trained_models/tiny_fast/checkpoint-64")
    """
    print(f"Loading base model: {base_model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print("✓ Model loaded successfully")
    return model, tokenizer


def load_rag_corpus(json_path: Path) -> List[dict]:
    """RAG용 코퍼스 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> set:
    # 아주 단순한 토크나이저: 공백/구두점 기준 분리
    return set(re.split(r"[\s\W]+", text.lower()))


def retrieve_context(query: str, corpus: List[dict], top_k: int = 3) -> List[str]:
    """간단한 bag-of-words 겹침 기반 검색 (외부 라이브러리 없이)"""
    q_tokens = _tokenize(query)
    scored: List[Tuple[float, str]] = []

    for item in corpus:
        doc_text = f"{item.get('player', '')} {item.get('npc', '')}"
        d_tokens = _tokenize(doc_text)
        if not d_tokens:
            continue
        overlap = len(q_tokens & d_tokens)
        if overlap > 0:
            scored.append((overlap / len(d_tokens), item.get("npc", "")))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


def build_prompt(player_input: str, contexts: List[str] | None = None) -> str:
    context_block = ""
    if contexts:
        bullets = "\n".join(f"- {c}" for c in contexts)
        context_block = f"### Memory (RAG):\n{bullets}\n\n"

    return (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"{context_block}"
        f"### Player:\n{player_input}\n\n"
        f"### NPC:\n"
    )


def generate_response(
    model,
    tokenizer,
    player_input: str,
    contexts: List[str] | None = None,
    max_tokens: int = 96,
):
    """NPC 응답 생성 (옵션으로 RAG 컨텍스트 사용)"""
    prompt = build_prompt(player_input, contexts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### NPC:" in full_response:
        npc_response = full_response.split("### NPC:")[-1].strip()
    elif "NPC:" in full_response:
        npc_response = full_response.split("NPC:")[-1].strip()
    else:
        npc_response = full_response.strip()

    return npc_response


def interactive_chat(model, tokenizer, corpus: List[dict] | None = None):
    """대화형 테스트"""
    print("\n" + "=" * 60)
    print("햄릿 NPC 대화 테스트 (RAG 포함)")
    print("'exit' 또는 'quit'를 입력하면 종료합니다.")
    print("=" * 60 + "\n")

    while True:
        player_input = input("Player: ").strip()

        if player_input.lower() in ["exit", "quit", "종료"]:
            print("대화를 종료합니다.")
            break

        if not player_input:
            continue

        contexts = retrieve_context(player_input, corpus) if corpus else None
        npc_response = generate_response(model, tokenizer, player_input, contexts)
        print(f"NPC: {npc_response}\n")


def batch_test(model, tokenizer, test_inputs: list, corpus: List[dict] | None = None):
    """배치 테스트"""
    print("\n" + "=" * 60)
    print("배치 테스트")
    print("=" * 60 + "\n")

    for i, player_input in enumerate(test_inputs, 1):
        print(f"[{i}/{len(test_inputs)}]")
        print(f"Player: {player_input}")

        contexts = retrieve_context(player_input, corpus) if corpus else None
        npc_response = generate_response(model, tokenizer, player_input, contexts)
        print(f"NPC: {npc_response}\n")


if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" #사용하실 모델, config.yaml이랑 같은거 쓰시면 됩니다.
    LORA_PATH = "trained_models/qwen15b_high_quality_ko" #저장된 폴더, config.yaml이랑 같은거 쓰시면 됩니다.
    USE_RAG = True

    # 경로
    here = Path(__file__).parent
    rag_json = here / "data" / "your_json_for_train.json"

    # 모델 로드
    model, tokenizer = load_trained_model(BASE_MODEL, LORA_PATH)

    # RAG 코퍼스 로드 (옵션)
    corpus = load_rag_corpus(rag_json) if USE_RAG and rag_json.exists() else None

    # 테스트 입력
    test_inputs = [
        "안녕",
        "당신은 누구죠?",
        "복수를 할 생각이야?",
        "사랑하는 사람은 있어?",
        "요즘 밤에 잠은 잘 자?",
        "고마워",
    ]

    # 배치 테스트
    batch_test(model, tokenizer, test_inputs, corpus)

    # 대화형 테스트
    interactive_chat(model, tokenizer, corpus)
```

> config.yaml
```
model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"
  trust_remote_code: true
  torch_dtype: "bfloat16"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 32
  lora_alpha: 64
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  output_dir: "../trained_models/qwen15b_high_quality_ko"
  num_train_epochs: 5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 0.0001
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  optim: "paged_adamw_8bit"
  bf16: true
  tf32: true
  logging_steps: 10
  save_strategy: "epoch"
  save_total_limit: 3
  dataloader_num_workers: 4
  group_by_length: true
  max_seq_length: 1024

data:
  train_data_path: "../data/hamlet_dialogues_ko_clean.json"
  prompt_template: |
    ### Instruction:
    You are Hamlet, a brooding but articulate game NPC. Answer briefly in English with a reflective tone.

    ### Player:
    {player}

    ### NPC:
    {npc}

misc:
  seed: 42
  logging_level: "info"
```
위 3개를 세팅하시면, 보다 편하게 관리하실 수 있습니다. 데이터 셋은 5000개 가량을 추천하며, 이 기본 셋업은 LoRA Fine Tuning + RAG로 매우 간단하게 설정한 코드들 입니다. 햄릿이라고 써 둔 이유는, 테스트할 때 가장 보편적으로 쓰기 좋은 설정이라 가져왔습니다.

#### 2. 빌드 -> 추론 순으로 실행하시면 됩니다.

# 문의

github issue 등으로 문의 주시면 확인합니다.