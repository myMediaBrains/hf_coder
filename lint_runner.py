# lint_runner.py (hf_coder)
#
# 제안이 적용된 직후 그 파일에 대해 자동으로 정적 검사를 돌린다.
# - .py -> ruff check
# - .js/.jsx/.ts/.tsx -> eslint (가장 가까운 package.json이 있는 폴더에서 npx로 실행)
#
# 둘 다 "설치돼 있으면 쓰고, 없으면 조용히 available=False로 알려주기만 하고
# 넘어간다" - hf_coder 서비스 자체가 이것 때문에 죽으면 안 된다 (지금까지
# 여러 번 겪었던 ModuleNotFoundError 패턴을 여기서는 반복하지 않기 위한 설계).
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _find_nearest_package_json(abs_path: Path, root: Path) -> Optional[Path]:
    """eslint 실행 기준 폴더를 찾는다 - 파일 위치에서 위로 올라가며 package.json 탐색."""
    current = abs_path.parent
    while True:
        if (current / "package.json").exists():
            return current
        if current == root or current.parent == current:
            return None
        current = current.parent


def _run_ruff(abs_path: Path) -> dict:
    if not shutil.which("ruff"):
        return {"tool": "ruff", "available": False, "detail": "ruff가 설치돼 있지 않습니다 (pip install ruff)"}
    try:
        proc = subprocess.run(
            ["ruff", "check", str(abs_path), "--output-format=concise"],
            capture_output=True, text=True, timeout=20,
        )
        # returncode==0이면 ruff가 문제를 못 찾은 것 - stdout에 "All checks
        # passed!" 같은 성공 메시지가 찍히는데, 이걸 issues로 세면 통과했는데도
        # "이슈 있음"으로 잘못 표시되는 버그가 있었다. returncode 기준으로 명확히 나눈다.
        if proc.returncode == 0:
            return {"tool": "ruff", "available": True, "ok": True, "issues": []}
        issues = [line for line in proc.stdout.splitlines() if line.strip()]
        return {"tool": "ruff", "available": True, "ok": False, "issues": issues}
    except Exception as e:
        return {"tool": "ruff", "available": True, "ok": False, "issues": [], "error": str(e)}


def _run_eslint(abs_path: Path, root: Path) -> dict:
    pkg_dir = _find_nearest_package_json(abs_path, root)
    if not pkg_dir:
        return {"tool": "eslint", "available": False, "detail": "package.json을 찾지 못했습니다."}
    if not shutil.which("npx"):
        return {"tool": "eslint", "available": False, "detail": "npx(node)를 찾지 못했습니다."}

    rel_to_pkg = str(abs_path.relative_to(pkg_dir))
    try:
        proc = subprocess.run(
            ["npx", "--no-install", "eslint", rel_to_pkg, "--format=compact"],
            capture_output=True, text=True, timeout=30, cwd=str(pkg_dir),
        )
        if proc.returncode == 0:
            return {"tool": "eslint", "available": True, "ok": True, "issues": []}
        issues = [line for line in proc.stdout.splitlines() if line.strip()]
        return {"tool": "eslint", "available": True, "ok": False, "issues": issues}
    except FileNotFoundError:
        return {"tool": "eslint", "available": False, "detail": "eslint 실행에 실패했습니다 (npx 확인 필요)."}
    except Exception as e:
        return {"tool": "eslint", "available": True, "ok": False, "issues": [], "error": str(e)}


def run_lint(root: Path, rel_path: str) -> dict:
    abs_path = root / rel_path
    suffix = abs_path.suffix

    if suffix == ".py":
        return _run_ruff(abs_path)
    if suffix in (".js", ".jsx", ".ts", ".tsx"):
        return _run_eslint(abs_path, root)

    return {"tool": None, "available": False, "detail": "이 파일 형식은 자동 린트 대상이 아닙니다."}
