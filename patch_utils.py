# patch_utils.py (hf_coder)
#
# Aider 방식 SEARCH/REPLACE 블록 파싱 + 적용. "파일 전체 재작성"보다 훨씬
# 안전하고 토큰도 적게 쓴다 - 모델은 바뀌는 부분만 정확히 써내면 되고,
# 적용 전에 "SEARCH 블록이 파일에 정확히 1번 있는지"를 검증해서 엉뚱한
# 곳에 적용되는 걸 막는다.
#
# 형식 (모델이 이 형식 그대로 응답에 포함시켜야 함):
#
#   relative/path/to/file.py
#   <<<<<<< SEARCH
#   (파일에 지금 있는 그대로의 코드, 몇 줄)
#   =======
#   (바꾼 뒤의 코드)
#   >>>>>>> REPLACE
#
# 한 응답에 여러 블록(여러 파일 포함 가능)이 있을 수 있다.
import re
from pathlib import Path
from typing import Callable, Optional

EDIT_BLOCK_RE = re.compile(
    r"([^\n]+)\n<<<<<<<\s*SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE",
    re.DOTALL,
)

PATCH_FORMAT_INSTRUCTIONS = """When proposing a code change, use this exact SEARCH/REPLACE block format \
(you may include multiple blocks, even across different files):

relative/path/to/file.py
<<<<<<< SEARCH
(the exact existing code, copied character-for-character including indentation - \
keep this block SHORT, just enough lines to uniquely identify the location)
=======
(the replacement code)
>>>>>>> REPLACE

Rules:
- The SEARCH block must match the current file content EXACTLY (including whitespace/indentation) \
so it can be found and replaced safely. If you are not fully certain of the exact existing text, \
do not propose an edit - just explain what should change in words instead.
- Keep each SEARCH block as short as possible (just the lines that need to change plus 1-2 lines \
of surrounding context), not the whole file.
- Only propose edits if you are confident they are correct. If unsure, explain your concerns \
in prose instead and do not include a SEARCH/REPLACE block."""


def extract_edits(response_text: str) -> tuple[str, list[dict]]:
    """LLM 응답에서 SEARCH/REPLACE 블록들을 뽑아내고, 남은 텍스트(리뷰 코멘트)를 반환."""
    edits = []

    def _consume(m: re.Match) -> str:
        raw_path = m.group(1).strip()
        # 모델이 백틱이나 "파일:" 같은 걸 앞에 덧붙이는 경우가 있어 방어적으로 정리.
        # .strip()만으로는 못 잡는 경우(정규식 [^\n]로 캡처된 라인 끝에 붙는 \r,
        # 유니코드 공백류 등)가 실제로 있었어서, 앞뒤 공백류 전부를 정규식으로
        # 한 번 더 확실히 제거한다 - 화면엔 똑같아 보여도 파일 못 찾는 원인이 됨.
        path = raw_path.strip("`").strip()
        path = re.sub(r"^(파일|file)\s*[:：]\s*", "", path, flags=re.IGNORECASE).strip()
        path = re.sub(r"^[\s\u00a0\u200b\ufeff]+|[\s\u00a0\u200b\ufeff]+$", "", path)
        edits.append({
            "path": path,
            "search": m.group(2),
            "replace": m.group(3),
        })
        return ""

    remaining = EDIT_BLOCK_RE.sub(_consume, response_text)
    return remaining.strip(), edits


def apply_edits(
    edits: list[dict],
    resolve_path_fn: Callable[[str], Path],
    mark_self_write_fn: Optional[Callable[[Path, str], None]] = None,
) -> list[dict]:
    """각 edit을 순서대로 적용. 실패한 edit이 있어도 나머지는 계속 시도하고,
    edit별 성공/실패 결과를 그대로 반환한다 (부분 적용을 사람이 눈으로 확인 가능하게)."""
    results = []
    for edit in edits:
        try:
            target = resolve_path_fn(edit["path"])
            if not target.is_file():
                # repr()로 보이지 않는 공백/특수문자까지 그대로 드러나게 해서,
                # "화면엔 똑같아 보이는데 실제 문자열은 다른" 케이스를 바로 진단할 수 있게 함.
                results.append({
                    **edit, "status": "error",
                    "detail": f"파일을 찾을 수 없습니다. 시도한 경로: {target} (원본 path 문자열: {edit['path']!r})",
                })
                continue

            original = target.read_text(encoding="utf-8", errors="ignore")
            count = original.count(edit["search"])

            if count == 0:
                results.append({**edit, "status": "error",
                                 "detail": "SEARCH 블록과 일치하는 부분을 파일에서 찾지 못했습니다 "
                                            "(파일이 이미 바뀌었을 수 있습니다)."})
                continue
            if count > 1:
                results.append({**edit, "status": "error",
                                 "detail": f"SEARCH 블록이 {count}번 일치합니다 - 안전하게 적용하려면 "
                                            "정확히 1번만 일치해야 합니다."})
                continue

            new_content = original.replace(edit["search"], edit["replace"], 1)
            target.write_text(new_content, encoding="utf-8")

            if mark_self_write_fn:
                mark_self_write_fn(target, new_content)

            results.append({**edit, "status": "applied"})
        except Exception as e:
            results.append({**edit, "status": "error", "detail": str(e)})

    return results


# 2026-08-13(신규): 특히 경량 모델(7b 등)에서, "SEARCH/REPLACE 블록을 만들겠다"고
# 텍스트로는 언급하면서 실제 <<<<<<< SEARCH 마커는 하나도 못 만드는 경우가 실제로
# 있었다 (예: "### SEARCH/REPLACE BLOCK 1"이라는 제목만 쓰고 내용이 빔). 이러면
# edits가 빈 리스트로 조용히 나와서, 사람이 "이 모델이 실패했다"는 걸 못 알아채고
# 그냥 "이번엔 제안이 없나 보다"고 넘어갈 위험이 있다 - 명시적으로 경고한다.
_BROKEN_PATCH_HINT_RE = re.compile(r"search\s*/\s*replace|search.?block|<<<<<<<|>>>>>>>", re.IGNORECASE)


def looks_like_failed_patch_attempt(raw_text: str, edits: list[dict]) -> bool:
    """edits가 비어있는데, 응답 텍스트 안에 SEARCH/REPLACE를 시도한 흔적(제목/언급)이
    있으면 True - "제안이 아예 없는 것"과 "제안하려다 형식을 못 지킨 것"을 구분한다."""
    if edits:
        return False
    return bool(_BROKEN_PATCH_HINT_RE.search(raw_text))