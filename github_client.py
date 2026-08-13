# github_client.py (hf_coder)
#
# GitHub REST API 래퍼. 두 가지 용도를 같은 함수로 공유한다:
# 1) "타 GitHub 탐색" - 사용자가 owner/repo를 직접 입력해서 읽기전용으로 훑어봄
# 2) "내 GitHub 확인" - CODE_PROJECT_ROOT의 git remote를 스스로 읽어 owner/repo를
#    자동으로 알아내고, 커밋&푸시 직후 실제로 반영됐는지 최신 커밋을 조회
#
# clone은 하지 않는다 - Trees/Contents/Commits API로 필요한 것만 그때그때 받는다.
# GITHUB_TOKEN(.env)이 있으면 시간당 5,000회, 없으면 60회로 제한된다(비공개
# 저장소는 토큰이 반드시 있어야 함).
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import requests

GITHUB_API = "https://api.github.com"


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(path: str, params: Optional[dict] = None) -> tuple[dict | list, dict]:
    """반환: (JSON 본문, 부가정보 - rate limit 등). 실패 시 GitHubAPIError를 던진다."""
    res = requests.get(f"{GITHUB_API}{path}", headers=_headers(), params=params, timeout=15)
    meta = {
        "rate_limit_remaining": res.headers.get("X-RateLimit-Remaining"),
        "rate_limit_limit": res.headers.get("X-RateLimit-Limit"),
    }
    if res.status_code == 404:
        raise GitHubAPIError(f"찾을 수 없습니다 (404): {path}")
    if res.status_code == 403 and res.headers.get("X-RateLimit-Remaining") == "0":
        raise GitHubAPIError("GitHub API 요청 한도를 초과했습니다. .env에 GITHUB_TOKEN을 설정하면 한도가 늘어납니다.")
    if not res.ok:
        raise GitHubAPIError(f"GitHub API 오류 ({res.status_code}): {res.text[:200]}")
    return res.json(), meta


class GitHubAPIError(Exception):
    pass


def repo_info(owner: str, repo: str) -> dict:
    data, meta = _get(f"/repos/{owner}/{repo}")
    return {
        "owner": owner, "repo": repo,
        "full_name": data.get("full_name"),
        "description": data.get("description"),
        "default_branch": data.get("default_branch"),
        "stars": data.get("stargazers_count"),
        "language": data.get("language"),
        "private": data.get("private"),
        "html_url": data.get("html_url"),
        "rate_limit": meta,
    }


def file_tree(owner: str, repo: str, branch: str) -> dict:
    data, meta = _get(f"/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": "1"})
    files = [
        {"path": item["path"], "type": item["type"], "size": item.get("size")}
        for item in data.get("tree", [])
        if item["type"] == "blob"
    ]
    return {"files": sorted(files, key=lambda f: f["path"]), "truncated": data.get("truncated", False), "rate_limit": meta}


def file_content(owner: str, repo: str, path: str, branch: str) -> dict:
    import base64
    data, meta = _get(f"/repos/{owner}/{repo}/contents/{path}", params={"ref": branch})
    if isinstance(data, list):
        raise GitHubAPIError("이 경로는 파일이 아니라 폴더입니다.")
    if data.get("encoding") != "base64":
        raise GitHubAPIError("지원하지 않는 인코딩입니다.")
    content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    return {"path": path, "content": content, "size": data.get("size"), "rate_limit": meta}


def recent_commits(owner: str, repo: str, branch: str, limit: int = 10) -> dict:
    data, meta = _get(f"/repos/{owner}/{repo}/commits", params={"sha": branch, "per_page": limit})
    commits = [
        {
            "sha": c["sha"][:10],
            "message": c["commit"]["message"].split("\n")[0],
            "author": c["commit"]["author"]["name"],
            "date": c["commit"]["author"]["date"],
            "html_url": c["html_url"],
        }
        for c in data
    ]
    return {"commits": commits, "rate_limit": meta}


def _parse_owner_repo_from_remote(remote_url: str) -> Optional[tuple[str, str]]:
    # git@github.com:owner/repo.git 또는 https://github.com/owner/repo.git 둘 다 처리
    patterns = [
        r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?/?$",
    ]
    for pat in patterns:
        m = re.search(pat, remote_url.strip())
        if m:
            return m.group(1), m.group(2)
    return None


def get_my_repo(project_root: Path) -> dict:
    """CODE_PROJECT_ROOT의 git remote origin을 읽어 owner/repo를 자동으로 알아낸다."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        raise GitHubAPIError(f"git remote 조회 실패: {e}")

    if proc.returncode != 0 or not proc.stdout.strip():
        raise GitHubAPIError("origin remote가 설정돼 있지 않습니다. `git remote add origin ...`을 먼저 해주세요.")

    parsed = _parse_owner_repo_from_remote(proc.stdout)
    if not parsed:
        raise GitHubAPIError(f"remote URL에서 owner/repo를 못 읽었습니다: {proc.stdout.strip()}")

    owner, repo = parsed

    # 로컬 현재 브랜치도 같이 알려준다 (원격 확인 시 어느 브랜치를 볼지 기본값으로 씀)
    branch_proc = subprocess.run(
        ["git", "-C", str(project_root), "branch", "--show-current"],
        capture_output=True, text=True, timeout=10,
    )
    local_branch = branch_proc.stdout.strip() or None

    info = repo_info(owner, repo)
    info["local_branch"] = local_branch
    return info
