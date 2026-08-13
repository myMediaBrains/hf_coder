# activity_tracker.py (hf_coder)
#
# 주의: 이건 hf_crawler의 실제 activity_tracker.py를 그대로 복사한 게 아니라,
# model_router.py가 기대하는 인터페이스(track_component 컨텍스트매니저)만
# 최소로 재구현한 버전이다. hf_crawler의 활동배지와 완전히 똑같은 동작을
# 원하면, hf_crawler/activity_tracker.py 파일을 그대로 복사해서 이 파일을
# 덮어써도 된다 (model_router.py 쪽 사용 방식은 동일하므로 그대로 호환됨).
import contextlib
import threading
import time

_lock = threading.Lock()
_components: dict[str, str] = {}


@contextlib.contextmanager
def track_component(name: str, status: str):
    """지금 이 컴포넌트가 뭘 하고 있는지 기록. with 블록이 끝나면 '대기 중'으로 되돌린다."""
    with _lock:
        _components[name] = status
    try:
        yield
    finally:
        with _lock:
            _components[name] = "대기 중"


def get_snapshot() -> dict:
    """지금 이 순간의 활동 상태 스냅샷. hf_crawler의 /stats/system과 동일한 형태를
    원하면 이 함수를 그대로 hf_coder의 /stats 엔드포인트에서 써도 된다."""
    with _lock:
        return {"requests": [], "components": dict(_components)}
