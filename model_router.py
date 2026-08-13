"""
model_router.py (hf_coder)
코딩분석 전용 Ollama 모델 라우팅 계층.

hf_crawler의 model_router.py에서 코딩분석과 무관한 프로파일(translate_*,
classify, rag_report, personalized_*, propose_taxonomy, summarize_readme,
analyze_repo_detail 등)을 전부 걷어내고 CODE 티어만 남긴 축소판이다.
구조(chat/chat_stream/achat_stream/start_stream)는 그대로 재사용했다.

지금은 CODE 티어 하나만 두지만, 나중에 "단순 질문은 7b, 복잡한 건 30b"
같은 2단계 자동 라우팅(CODE_LIGHT/CODE_HEAVY)을 추가하고 싶으면 이 파일
안에서만 확장하면 된다 - hf_crawler 쪽은 이 파일의 존재 자체를 모른다.
"""

import asyncio
import logging
from enum import Enum
from typing import AsyncGenerator, Iterator, Optional

import threading

import ollama
import activity_tracker

logger = logging.getLogger(__name__)

_active_lock = threading.Lock()
_active_count = 0


def is_generating() -> bool:
    """지금 이 순간 실제로 생성 중인 요청이 있는지 (GPU 실사용 여부 추정에 사용)."""
    with _active_lock:
        return _active_count > 0


def _mark_start():
    global _active_count
    with _active_lock:
        _active_count += 1


def _mark_end():
    global _active_count
    with _active_lock:
        _active_count = max(0, _active_count - 1)


class ModelTier(str, Enum):
    CODE = "code"            # 코딩분석 채팅 전용 - qwen3-coder:30b-a3b (MoE), 온디맨드 로드
    CODE_LIGHT = "code_light"  # 2026-08-13: "빠른 모드" 전용 - qwen2.5-coder:14b-instruct-q4_K_M, 상시 상주


TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.CODE: "qwen3-coder:30b-a3b-q4_K_M",
    ModelTier.CODE_LIGHT: "qwen2.5-coder:14b-instruct-q4_K_M",
}

# 10분: 대화가 이어지는 동안은 재로드 비용 없게, 대화가 끊기면 반납.
# CODE_LIGHT는 LIGHT(9b) 티어처럼 상시 상주(항상 준비된 상태로 켜자마자 응답).
# 2026-08-13: 7b -> 14b-instruct-q4_K_M으로 교체 실험 중 - 7b가 SEARCH/REPLACE
# 형식을 못 지키거나 없는 코드를 지어내는 문제가 실제로 있어서, 품질을 좀 더
# 확보하려고 시도. 다만 14b(~8~9GB)는 7b(~4.7GB)보다 상주 비용이 커서,
# LIGHT+CODE_LIGHT 합산 상시 메모리가 약 15~16GB로 늘어난다 - 여전히 CODE(30b,
# 온디맨드 ~19GB) 상시 상주보다는 가볍지만, 예전보다는 기본 부담이 커졌음.
# 적다 - "빠른 모드"를 켜자마자 콜드스타트 없이 바로 답하게 하려는 목적.
TIER_KEEP_ALIVE: dict[ModelTier, str | int] = {
    ModelTier.CODE: "10m",
    ModelTier.CODE_LIGHT: -1,
}

TASK_PROFILES: dict[str, dict] = {
    "code_analysis": {
        # 코드는 창의성보다 정확성이 중요하므로 temperature를 낮게, 선택된
        # 소스 파일 여러 개 + 대화 이력을 담아야 해서 num_ctx는 넉넉히 잡는다.
        "tier": ModelTier.CODE,
        "options": {
            "temperature": 0.15,
            "num_predict": 3072,
            "num_ctx": 16384,
            "presence_penalty": 0.1,
            "repeat_penalty": 1.15,
        },
    },
    "code_analysis_light": {
        # 2026-08-13: "⚡ 빠른 모드" 전용. 단일 파일 리뷰처럼 lint가 실수를
        # 상당 부분 보완해줄 수 있는 간단한 요청에 씀 - 정확성보다 응답
        # 속도/메모리 여유가 중요할 때 사용자가 직접 켜는 토글.
        # 2026-08-13(개정): 14b 모델이 num_predict 끝까지 같은 문단을 무한
        # 반복하는 증상이 실제로 있었다 - repeat_penalty를 명시적으로 높이고
        # presence_penalty도 올려서 반복을 더 강하게 억제한다.
        "tier": ModelTier.CODE_LIGHT,
        "options": {
            "temperature": 0.2,
            "num_predict": 2048,
            "num_ctx": 8192,
            "presence_penalty": 0.4,
            "repeat_penalty": 1.3,
            "repeat_last_n": 256,
        },
    },
    "review_diff": {
        # 2026-08-13: VS Code 저장 감지 자동리뷰 전용. diff + 파일 전체 내용을
        # 담아야 해서 num_ctx는 code_analysis보다 더 넉넉히, 그러나 리뷰
        # 코멘트 자체는 짧게 - 수정 제안은 파일 전체를 다시 써야 하므로
        # num_predict는 넉넉히 잡는다(원본 파일 길이만큼 필요할 수 있음).
        "tier": ModelTier.CODE,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
            "num_ctx": 20480,
            "presence_penalty": 0.1,
            "repeat_penalty": 1.15,
        },
    },
    "architect_plan": {
        # 2026-08-13: Architect 모드 1단계(설계). SEARCH/REPLACE 형식을 전혀
        # 신경 안 쓰고 "무엇을 왜 바꿔야 하는지"에만 집중해서 추론한다 -
        # temperature를 code_analysis보다 살짝 높여서 여러 접근을 더 자유롭게
        # 검토하게 하고, num_predict도 넉넉히 줘서 계획을 충분히 풀어 쓰게 한다.
        # 지금은 code_analysis와 같은 CODE 티어(같은 모델)를 쓰지만, 나중에
        # 다른 모델을 쓰고 싶으면 이 tier 값만 바꾸면 된다.
        "tier": ModelTier.CODE,
        "options": {
            "temperature": 0.3,
            "num_predict": 2048,
            "num_ctx": 16384,
            "presence_penalty": 0.2,
            "repeat_penalty": 1.15,
        },
    },
    "agent_loop": {
        # 2026-08-13: 툴콜링 에이전트 루프 전용. 도구 이름/스키마를 정확히
        # 지켜야 하므로 temperature는 낮게. 도구 결과가 반복적으로 쌓이면서
        # 대화가 길어지므로 num_ctx를 가장 넉넉하게 잡는다.
        # 2026-08-13(개정): read_file 청크가 8000자->16000자로 커진 만큼
        # num_ctx도 같이 올림 (메모리 여유 보고 필요하면 더 조정 가능).
        "tier": ModelTier.CODE,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": 32768,
            "presence_penalty": 0.1,
            "repeat_penalty": 1.15,
        },
    },
    "agent_loop_light": {
        # 2026-08-13(신규): 에이전트 모드에서 qwen2.5-coder:14b(CODE_LIGHT)를
        # 써보는 실험용 프로필. 30b보다 빠를 걸 기대하지만, 툴콜링을 정확히
        # 못 지키거나 도구 이름/인자를 잘못 만들 위험도 있어 비교 실험 목적.
        "tier": ModelTier.CODE_LIGHT,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": 16384,
            "presence_penalty": 0.2,
            "repeat_penalty": 1.2,
        },
    },
}


def model_for_task(task: str) -> str:
    """task 프로필이 실제로 사용하는 모델 태그를 반환한다 (CodeChatMessage.model_used 기록용)."""
    if task not in TASK_PROFILES:
        raise ValueError(
            f"등록되지 않은 task 프로필입니다: '{task}'. "
            f"사용 가능: {list(TASK_PROFILES.keys())}"
        )
    return TIER_MODELS[TASK_PROFILES[task]["tier"]]


def _build_request(
    task: str,
    messages: list[dict],
    extra_options: Optional[dict] = None,
) -> dict:
    if task not in TASK_PROFILES:
        raise ValueError(
            f"등록되지 않은 task 프로필입니다: '{task}'. "
            f"사용 가능: {list(TASK_PROFILES.keys())}"
        )

    profile = TASK_PROFILES[task]
    tier = profile["tier"]
    model = TIER_MODELS[tier]
    options = {**profile["options"], **(extra_options or {})}

    return {
        "model": model,
        "messages": messages,
        "options": options,
        "think": False,
        "keep_alive": TIER_KEEP_ALIVE[tier],
    }


def chat(
    task: str,
    messages: list[dict],
    extra_options: Optional[dict] = None,
) -> str:
    request = _build_request(task, messages, extra_options)
    logger.info(
        f"[model_router] task={task} model={request['model']} "
        f"keep_alive={request['keep_alive']}"
    )
    _mark_start()
    with activity_tracker.track_component("LLM 모델 라우터", f"{task} 처리 중 ({request['model']})"):
        try:
            response = ollama.chat(**request)
            return response["message"]["content"].strip()
        finally:
            _mark_end()


def chat_stream(
    task: str,
    messages: list[dict],
    extra_options: Optional[dict] = None,
) -> Iterator[dict]:
    request = _build_request(task, messages, extra_options)
    request["stream"] = True
    logger.info(
        f"[model_router] (stream) task={task} model={request['model']} "
        f"keep_alive={request['keep_alive']}"
    )
    return ollama.chat(**request)


async def achat_stream(
    task: str,
    messages: list[dict],
    extra_options: Optional[dict] = None,
) -> AsyncGenerator[dict, None]:
    queue, sentinel = await start_stream(task, messages, extra_options)
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def start_stream(
    task: str,
    messages: list[dict],
    extra_options: Optional[dict] = None,
) -> tuple[asyncio.Queue, object]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _producer():
        _mark_start()
        model_name = TIER_MODELS[TASK_PROFILES[task]["tier"]]
        with activity_tracker.track_component("LLM 모델 라우터", f"{task} 스트리밍 중 ({model_name})"):
            try:
                for chunk in chat_stream(task, messages, extra_options):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                _mark_end()
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    logger.info(f"[model_router] (start_stream) task={task}")
    threading.Thread(target=_producer, daemon=True, name=f"model_router-stream-{task}").start()
    return queue, SENTINEL


def chat_with_tools(
    task: str,
    messages: list[dict],
    tools: list[dict],
    extra_options: Optional[dict] = None,
) -> dict:
    """
    2026-08-13: 툴콜링 에이전트 루프 전용. 스트리밍 없이 한 번에 응답을 받아서
    tool_calls가 있는지 확인하는 용도라 stream=False로 호출한다 (에이전트 루프
    자체는 여러 번 반복 호출되는데, 매 반복을 다 스트리밍하면 오히려 프론트에서
    다루기 복잡해지니, 최종 답변만 main.py 쪽에서 별도로 스트리밍한다).
    반환값은 ollama 원본 응답(dict-like) 그대로 - message.tool_calls / message.content 사용.
    """
    request = _build_request(task, messages, extra_options)
    request["tools"] = tools
    logger.info(f"[model_router] (tools) task={task} model={request['model']} tools={len(tools)}개")
    _mark_start()
    with activity_tracker.track_component("LLM 모델 라우터", f"{task} 도구 호출 판단 중 ({request['model']})"):
        try:
            return ollama.chat(**request)
        finally:
            _mark_end()