"""
server.py
나스닥 RAG 검색 서버 — Railway 배포용
환경변수로 API 키 관리 (코드에 키를 직접 입력하지 않음)
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ─────────────────────────────────────────
# 환경변수에서 설정 읽기 (Railway 대시보드에서 입력)
OPENAI_KEY  = os.environ.get("OPENAI_KEY", "")
CLAUDE_KEY  = os.environ.get("CLAUDE_KEY", "")
DB_FOLDER   = os.environ.get("DB_FOLDER", "chroma_db")
PORT        = int(os.environ.get("PORT", 8765))   # Railway가 자동으로 포트 배정
TOP_K       = 5
# ─────────────────────────────────────────


def check_requirements():
    """서버 시작 전 필수 항목 점검"""
    ok = True

    if not OPENAI_KEY or not OPENAI_KEY.startswith("sk-"):
        print("❌ OPENAI_KEY 환경변수가 없거나 올바르지 않아요.")
        ok = False

    if not CLAUDE_KEY or not CLAUDE_KEY.startswith("sk-ant-"):
        print("❌ CLAUDE_KEY 환경변수가 없거나 올바르지 않아요.")
        ok = False

    if not os.path.exists(DB_FOLDER):
        print(f"❌ DB 폴더 '{DB_FOLDER}' 가 없어요. chroma_db 폴더가 업로드됐는지 확인하세요.")
        ok = False

    try:
        import openai, chromadb, anthropic
    except ImportError as e:
        print(f"❌ 라이브러리 없음: {e}")
        ok = False

    if not ok:
        sys.exit(1)

    print("✅ 모든 설정 확인 완료")


def search_docs(query):
    """쿼리 → 임베딩 → ChromaDB 유사도 검색"""
    import openai
    import chromadb

    client_openai = openai.OpenAI(api_key=OPENAI_KEY)
    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = chroma_client.get_collection("nasdaq_docs")

    response = client_openai.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "content":  doc,
            "filename": meta.get("filename", ""),
            "filepath": meta.get("filepath", ""),
            "preview":  meta.get("preview", doc[:200]),
            "score":    round(1 - dist, 3)
        })
    return docs


def generate_answer(query, docs):
    """검색된 문서 → Claude API → 답변 생성"""
    import anthropic

    client = anthropic.Anthropic(api_key=CLAUDE_KEY)

    context = ""
    for i, doc in enumerate(docs):
        context += f"\n[출처 {i+1}: {doc['filename']}]\n{doc['content']}\n"

    system_prompt = """당신은 나스닥 경제 콘텐츠 전문 검색 어시스턴트입니다.
제공된 출처 문서만을 기반으로 답변하세요.
출처에 없는 내용은 추측하지 말고 "해당 내용은 수집된 콘텐츠에서 찾을 수 없습니다"라고 답변하세요.
답변은 한국어로, 핵심을 먼저 말하고 근거를 설명하는 방식으로 작성하세요."""

    user_message = f"""질문: {query}

참고 문서:
{context}

위 문서를 바탕으로 질문에 답변해주세요."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    return response.content[0].text


class RAGHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # 기본 로그 억제

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """CORS preflight 처리"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """health check — Railway가 서버 살아있는지 확인용"""
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_error(404, "경로를 찾을 수 없어요.")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/search":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                data   = json.loads(body.decode("utf-8"))
                query  = data.get("query", "").strip()

                if not query:
                    self._send_error(400, "질문을 입력해주세요.")
                    return

                print(f"🔍 검색: {query}")

                docs   = search_docs(query)
                answer = generate_answer(query, docs)

                result = {
                    "answer": answer,
                    "sources": [
                        {
                            "filename": d["filename"],
                            "filepath": d["filepath"],
                            "preview":  d["preview"][:300],
                            "score":    d["score"]
                        }
                        for d in docs
                    ]
                }

                self._send_json(200, result)
                print(f"✅ 답변 완료 (출처 {len(docs)}개)")

            except json.JSONDecodeError:
                self._send_error(400, "요청 형식이 올바르지 않아요.")
            except Exception as e:
                print(f"❌ 오류: {e}")
                self._send_error(500, f"서버 오류: {str(e)}")
        else:
            self._send_error(404, "경로를 찾을 수 없어요.")

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        self._send_json(code, {"error": message})


def main():
    check_requirements()
    # Railway 배포: 0.0.0.0 으로 바인딩해야 외부 접속 가능
    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    print(f"\n{'='*50}")
    print(f"  🚀 RAG 서버 실행 중 (Railway 배포용)")
    print(f"{'='*50}")
    print(f"  포트: {PORT}")
    print(f"  DB:   {os.path.abspath(DB_FOLDER)}")
    print(f"  health check: GET /health\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료했어요.")


if __name__ == "__main__":
    main()
