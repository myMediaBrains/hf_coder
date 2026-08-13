# hf_coder

hf_crawler의 "💻 코딩분석" 기능을 독립시킨 별도 서비스입니다. hf_crawler와는
완전히 다른 프로세스/venv/DB로 돌고, HTTP로만 통신합니다.

## 처음 실행하기

```bash
cd hf_coder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env 열어서 CODE_PROJECT_ROOT를 실제 프로젝트 경로로 확인/수정

uvicorn main:app --reload --port 8100
```

기동 후 `http://localhost:8100/health` 로 정상 기동 확인, `http://localhost:8100/codeanalysis/files` 로 파일트리 확인.

## hf-frontend 쪽 연결

`CodeAnalysisChat.jsx`의 `API_URL`이 `VITE_CODE_API_URL`(기본값 `http://localhost:8100`)을 보도록
이미 수정되어 있습니다. hf-frontend의 `.env`에 아래 한 줄만 추가하면 됩니다.

```
VITE_CODE_API_URL=http://localhost:8100
```

## 다음에 확장할 만한 것

- Docker 기반 샌드박스: GitHub/HF에서 받아온 오픈소스를 격리 실행하는 기능
- 자기확장 워크플로: 에이전트가 diff를 제안 → 격리 테스트 → 사람 승인 → git commit
- CODE_LIGHT(qwen2.5-coder:7b) / CODE_HEAVY(qwen3-coder:30b-a3b) 2단계 자동 라우팅
- codebase-memory: 코드 임베딩 기반 자동 관련 파일 검색(RAG)

전부 이 서비스 안에서만 손보면 되고, hf_crawler는 건드릴 필요가 없습니다.
