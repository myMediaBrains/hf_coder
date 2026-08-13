# sandbox.py (hf_coder)
#
# GitHub 오픈소스를 clone해서 Docker 컨테이너 안에서만 격리 실행한다.
# 로컬 프로젝트(CODE_PROJECT_ROOT)와는 완전히 별개 영역 - 여기서 무슨 일이
# 나도 실제 작업 중인 코드에는 영향이 없다.
#
# 안전 원칙:
# - 컨테이너는 매번 새로(--rm) - 이전 실행의 흔적이 다음 실행에 안 남음
# - 네트워크 기본 차단(--network none) - 받아온 코드가 뭔가를 몰래 외부로
#   보내는 걸 원천 차단. 정말 필요하면(예: pip install) 명시적으로만 허용.
# - CPU/메모리/실행시간 제한 - 무한루프나 메모리 폭탄 방지
# - 코드는 읽기전용으로 마운트 - 컨테이너 안에서 원본을 못 건드림
# - 한 번에 하나만 실행 - 맥북 메모리가 한정적이라 전역 락으로 직렬화
# - docker 자체가 없어도 서비스가 안 죽고 "사용불가"로만 표시
import logging
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.getenv("SANDBOX_WORKSPACE", "/tmp/hf_coder_sandbox"))
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

_run_lock = threading.Lock()  # 한 번에 하나만 실행

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300
MEMORY_LIMIT = "512m"
CPU_LIMIT = "1"

# 언어 감지 -> 베이스 이미지. 필요할 때만 pull되므로 미리 다 받아둘 필요 없음.
_LANGUAGE_MARKERS: list[tuple[str, str]] = [
    ("requirements.txt", "python:3.12-slim"),
    ("pyproject.toml", "python:3.12-slim"),
    ("setup.py", "python:3.12-slim"),
    ("package.json", "node:20-slim"),
    ("go.mod", "golang:1.22-alpine"),
    ("Cargo.toml", "rust:1.78-slim"),
]
FALLBACK_IMAGE = "ubuntu:22.04"


def is_available() -> bool:
    return shutil.which("docker") is not None


def _run_git(args: list[str], cwd: Optional[str] = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, capture_output=True, text=True, timeout=timeout, cwd=cwd)


def clone_repo(owner: str, repo: str, branch: Optional[str] = None) -> dict:
    """GitHub 저장소를 얕은 clone(--depth 1)으로 임시 워크스페이스에 받는다."""
    workspace_id = uuid.uuid4().hex[:12]
    dest = WORKSPACE_ROOT / workspace_id

    url = f"https://github.com/{owner}/{repo}.git"
    args = ["clone", "--depth", "1"]
    if branch:
        args += ["--branch", branch]
    args += [url, str(dest)]

    proc = _run_git(args, timeout=120)
    if proc.returncode != 0:
        raise SandboxError(f"clone 실패: {proc.stderr.strip()[:500]}")

    image = detect_image(dest)
    return {
        "workspace_id": workspace_id,
        "path": str(dest),
        "detected_image": image,
    }


def detect_image(workspace_dir: Path) -> str:
    for marker_file, image in _LANGUAGE_MARKERS:
        if (workspace_dir / marker_file).exists():
            return image
    return FALLBACK_IMAGE


def cleanup_workspace(workspace_id: str) -> bool:
    dest = WORKSPACE_ROOT / workspace_id
    if not dest.exists():
        return False
    shutil.rmtree(dest, ignore_errors=True)
    return True


class SandboxError(Exception):
    pass


def run_in_sandbox(
    workspace_id: str,
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    allow_network: bool = False,
) -> dict:
    """
    격리 컨테이너에서 명령을 실행하고 결과를 반환한다.
    한 번에 하나만 실행되도록 전역 락을 건다 - 대기 중인 다른 요청은
    이 실행이 끝날 때까지 블록된다(맥북 메모리 여유가 크지 않아서 동시
    실행을 허용하지 않는 게 안전).
    """
    if not is_available():
        raise SandboxError("Docker가 설치돼 있지 않습니다. https://docker.com 에서 설치해주세요.")

    workspace_dir = WORKSPACE_ROOT / workspace_id
    if not workspace_dir.exists():
        raise SandboxError("워크스페이스를 찾을 수 없습니다. 먼저 clone을 실행해주세요.")

    timeout = min(max(1, timeout), MAX_TIMEOUT)
    image = detect_image(workspace_dir)

    docker_args = [
        "docker", "run", "--rm",
        "--memory", MEMORY_LIMIT,
        "--cpus", CPU_LIMIT,
        "-v", f"{workspace_dir}:/workspace:ro",
        "-w", "/workspace",
    ]
    if not allow_network:
        docker_args += ["--network", "none"]
    docker_args += [image, "sh", "-c", command]

    with _run_lock:
        logger.info(f"[sandbox] 실행 시작: {workspace_id} image={image} network={'허용' if allow_network else '차단'}")
        try:
            proc = subprocess.run(docker_args, capture_output=True, text=True, timeout=timeout)
            return {
                "workspace_id": workspace_id,
                "image": image,
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-20000:],  # 너무 길면 뒤쪽만 (앞부분보다 결과/에러가 뒤에 몰림)
                "stderr": proc.stderr[-20000:],
                "timed_out": False,
                "network_allowed": allow_network,
            }
        except subprocess.TimeoutExpired:
            return {
                "workspace_id": workspace_id,
                "image": image,
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": f"실행이 {timeout}초를 넘어 강제 종료되었습니다.",
                "timed_out": True,
                "network_allowed": allow_network,
            }
        finally:
            logger.info(f"[sandbox] 실행 종료: {workspace_id}")
