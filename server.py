"""
server.py
나스닥 RAG 검색 서버 — Railway 배포용
Volume 마운트 환경: chroma_db Volume에 영구 저장
"""

import os
import sys
import json
import glob
import time
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ─────────────────────────────────────────
OPENAI_KEY  = os.environ.get("OPENAI_KEY", "")
CLAUDE_KEY  = os.environ.get("CLAUDE_KEY", "")   # fallback용 (없어도 됨)
DB_FOLDER   = os.environ.get("DB_FOLDER", "chroma_db")
MD_FOLDER   = os.environ.get("MD_FOLDER", "md_files")
PORT        = int(os.environ.get("PORT", 8765))
TOP_K       = 5
CHUNK_SIZE  = 1000
BATCH_SIZE  = 50
# ─────────────────────────────────────────


def check_keys():
    if not OPENAI_KEY or not OPENAI_KEY.startswith("sk-"):
        print("❌ OPENAI_KEY 환경변수가 없거나 올바르지 않아요.")
        sys.exit(1)
    if not CLAUDE_KEY:
        print("⚠️  CLAUDE_KEY 환경변수 없음 — 요청에서 claude_key를 받아서 사용합니다.")
    print("✅ API 키 확인 완료")


def collection_exists():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_FOLDER)
        client.get_collection("nasdaq_docs")
        return True
    except Exception:
        return False


# ✅ 추가: MD 파일 상단 frontmatter에서 title, date, source 추출
def parse_frontmatter(content):
    title = ""
    date  = ""
    source = ""

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if match:
        fm = match.group(1)

        t = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
        if t:
            title = t.group(1).strip().strip('"').strip("'")

        d = re.search(r'^date:\s*(.+?)\s*$', fm, re.MULTILINE)
        if d:
            date = d.group(1).strip()

        s = re.search(r'^source:\s*(.+?)\s*$', fm, re.MULTILINE)
        if s:
            source = s.group(1).strip()

    return title, date, source


def get_indexed_filenames():
    """DB에 이미 저장된 파일명 목록 반환"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_FOLDER)
        collection = client.get_collection("nasdaq_docs")
        result = collection.get(include=["metadatas"])
        return set(m["filename"] for m in result["metadatas"])
    except Exception:
        return set()


def build_incremental():
    """새 md 파일만 임베딩해서 기존 DB에 추가"""
    import openai
    import chromadb

    md_files = glob.glob(os.path.join(MD_FOLDER, "**", "*.md"), recursive=True)
    if not md_files:
        print("❌ md_files 폴더에 .md 파일이 없어요.")
        return

    indexed = get_indexed_filenames()
    new_files = [f for f in md_files if os.path.basename(f) not in indexed]

    if not new_files:
        print(f"✅ 새 파일 없음 — 증분 업데이트 건너뜀 ({len(indexed)}개 기존 파일)")
        return

    print(f"🔄 증분 업데이트: {len(new_files)}개 새 파일 발견 (기존 {len(indexed)}개)")

    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = chroma_client.get_collection("nasdaq_docs")

    existing = collection.get(include=[])
    base_idx = len(existing["ids"]) if existing["ids"] else 0

    documents, metadatas, ids = [], [], []
    for i, filepath in enumerate(new_files):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if not content:
                continue

            filename = os.path.basename(filepath)
            rel_path = os.path.relpath(filepath, MD_FOLDER)
            title, date, source = parse_frontmatter(content)

            if len(content) > CHUNK_SIZE:
                chunks = [content[j:j+CHUNK_SIZE] for j in range(0, len(content), CHUNK_SIZE)]
                for k, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append(chunk)
                        metadatas.append({
                            "filename": filename,
                            "filepath": rel_path,
                            "chunk":    k,
                            "preview":  content[:200],
                            "title":    title,
                            "date":     date,
                            "source":   source,
                        })
                        ids.append(f"inc_{base_idx + i}_{k}")
            else:
                documents.append(content)
                metadatas.append({
                    "filename": filename,
                    "filepath": rel_path,
                    "chunk":    0,
                    "preview":  content[:200],
                    "title":    title,
                    "date":     date,
                    "source":   source,
                })
                ids.append(f"inc_{base_idx + i}_0")
        except Exception as e:
            print(f"  ⚠️ 읽기 실패: {filepath} → {e}")

    if not documents:
        print("⚠️ 추가할 유효한 청크 없음")
        return

    print(f"✅ {len(documents)}개 청크 증분 임베딩 시작")
    client_openai = openai.OpenAI(api_key=OPENAI_KEY)
    start_time = time.time()
    success = 0

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[batch_start:batch_start+BATCH_SIZE]
        batch_meta = metadatas[batch_start:batch_start+BATCH_SIZE]
        batch_ids  = ids[batch_start:batch_start+BATCH_SIZE]

        valid = [(d, m, i) for d, m, i in zip(batch_docs, batch_meta, batch_ids) if d.strip()]
        if not valid:
            continue
        v_docs, v_meta, v_ids = zip(*valid)

        try:
            response = client_openai.embeddings.create(
                input=list(v_docs),
                model="text-embedding-3-small"
            )
            embeddings = [item.embedding for item in response.data]
            collection.add(
                documents=list(v_docs),
                embeddings=embeddings,
                metadatas=list(v_meta),
                ids=list(v_ids)
            )
            success += len(v_docs)
            pct = min(100, int((batch_start + BATCH_SIZE) / len(documents) * 100))
            print(f"  진행: {pct}% ({success}/{len(documents)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️ 배치 오류: {e}")
            time.sleep(2)

    elapsed = time.time() - start_time
    print(f"✅ 증분 임베딩 완료! {success}개 추가 / {elapsed:.0f}초")


def build_db():
    import openai
    import chromadb

    print(f"\n📂 DB 생성 시작 (md_files → chroma_db)")

    md_files = glob.glob(os.path.join(MD_FOLDER, "**", "*.md"), recursive=True)
    if not md_files:
        print(f"❌ md_files 폴더에 .md 파일이 없어요.")
        sys.exit(1)
    print(f"✅ .md 파일 {len(md_files)}개 발견")

    documents, metadatas, ids = [], [], []
    for i, filepath in enumerate(md_files):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if not content:
                continue

            filename = os.path.basename(filepath)
            rel_path = os.path.relpath(filepath, MD_FOLDER)

            # ✅ frontmatter 파싱
            title, date, source = parse_frontmatter(content)

            if len(content) > CHUNK_SIZE:
                chunks = [content[j:j+CHUNK_SIZE] for j in range(0, len(content), CHUNK_SIZE)]
                for k, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append(chunk)
                        metadatas.append({
                            "filename": filename,
                            "filepath": rel_path,
                            "chunk":    k,
                            "preview":  content[:200],
                            "title":    title,
                            "date":     date,
                            "source":   source,
                        })
                        ids.append(f"{i}_{k}")
            else:
                documents.append(content)
                metadatas.append({
                    "filename": filename,
                    "filepath": rel_path,
                    "chunk":    0,
                    "preview":  content[:200],
                    "title":    title,
                    "date":     date,
                    "source":   source,
                })
                ids.append(f"{i}_0")
        except Exception as e:
            print(f"  ⚠️ 읽기 실패: {filepath} → {e}")

    print(f"✅ 총 {len(documents)}개 청크 준비")

    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)

    try:
        chroma_client.delete_collection("nasdaq_docs")
        print("🗑️ 기존 컬렉션 삭제")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="nasdaq_docs",
        metadata={"hnsw:space": "cosine"}
    )

    client_openai = openai.OpenAI(api_key=OPENAI_KEY)
    print(f"🔄 임베딩 변환 중... (총 {len(documents)}개)")
    start_time = time.time()
    success = 0

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[batch_start:batch_start+BATCH_SIZE]
        batch_meta = metadatas[batch_start:batch_start+BATCH_SIZE]
        batch_ids  = ids[batch_start:batch_start+BATCH_SIZE]

        valid = [(d, m, i) for d, m, i in zip(batch_docs, batch_meta, batch_ids) if d.strip()]
        if not valid:
            continue
        v_docs, v_meta, v_ids = zip(*valid)

        try:
            response = client_openai.embeddings.create(
                input=list(v_docs),
                model="text-embedding-3-small"
            )
            embeddings = [item.embedding for item in response.data]
            collection.add(
                documents=list(v_docs),
                embeddings=embeddings,
                metadatas=list(v_meta),
                ids=list(v_ids)
            )
            success += len(v_docs)
            pct = min(100, int((batch_start + BATCH_SIZE) / len(documents) * 100))
            print(f"  진행: {pct}% ({success}/{len(documents)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️ 배치 오류: {e}")
            time.sleep(2)

    elapsed = time.time() - start_time
    print(f"✅ 임베딩 완료! {success}개 / {elapsed:.0f}초")


def init():
    check_keys()

    try:
        import openai, chromadb, anthropic
    except ImportError as e:
        print(f"❌ 라이브러리 없음: {e}")
        sys.exit(1)

    # ✅ FORCE_REBUILD=true 환경변수 있으면 DB 강제 재생성 (1회용)
    force_rebuild = os.environ.get("FORCE_REBUILD", "").lower() == "true"

    if force_rebuild:
        print("🔄 FORCE_REBUILD 감지 → DB 강제 재생성 시작")
        if not os.path.exists(MD_FOLDER):
            print(f"❌ md_files 폴더가 없어요.")
            sys.exit(1)
        build_db()
        print("✅ DB 재생성 완료 — Railway에서 FORCE_REBUILD 환경변수를 삭제해주세요!")
    elif collection_exists():
        print(f"✅ DB 확인 완료 (컬렉션 정상) — 증분 업데이트 확인 중")
        build_incremental()

    else:
        print("⚠️ 컬렉션 없음 → DB 새로 생성")
        if not os.path.exists(MD_FOLDER):
            print(f"❌ md_files 폴더가 없어요.")
            sys.exit(1)
        build_db()

    print("✅ 초기화 완료\n")


def search_docs(query):
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
            "score":    round(1 - dist, 3),
            "title":    meta.get("title", ""),
            "date":     meta.get("date", ""),
            "source":   meta.get("source", ""),
        })
    return docs


def generate_answer(query, docs, claude_key):
    import anthropic

    key_to_use = claude_key if claude_key else CLAUDE_KEY
    if not key_to_use:
        raise ValueError("Claude API 키가 없어요. 화면에서 키를 입력해주세요.")

    client = anthropic.Anthropic(api_key=key_to_use)
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n[출처 {i+1}: {doc['filename']}]\n{doc['content']}\n"

    system_prompt = """당신은 나스닥 경제 콘텐츠 전문 검색 어시스턴트입니다.
제공된 출처 문서만을 기반으로 답변하세요.
출처에 없는 내용은 추측하지 말고 "해당 내용은 수집된 콘텐츠에서 찾을 수 없습니다"라고 답변하세요.
답변은 한국어로, 핵심을 먼저 말하고 근거를 설명하는 방식으로 작성하세요.
마크다운 문법(##, **, - 등)을 사용해도 됩니다.
답변 본문에서 특정 내용의 근거가 된 출처는 해당 문장 끝에 <sup>[출처 N]</sup> 형태로 작게 표시하세요. N은 참고 문서의 번호입니다."""

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
        pass

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if path == "/" or path == "/search_app.html":
            html_path = os.path.join(os.path.dirname(__file__), "search_app.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(html_content))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(html_content)
            else:
                self._send_error(404, "search_app.html 파일을 찾을 수 없어요.")
            return

        self._send_error(404, "경로를 찾을 수 없어요.")

    def do_POST(self):
        if urlparse(self.path).path == "/search":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                data   = json.loads(body.decode("utf-8"))
                query  = data.get("query", "").strip()
                claude_key = data.get("claude_key", "").strip()

                if not query:
                    self._send_error(400, "질문을 입력해주세요.")
                    return

                key_to_use = claude_key if claude_key else CLAUDE_KEY
                if not key_to_use or not key_to_use.startswith("sk-ant-"):
                    self._send_error(400, "Claude API 키가 없거나 형식이 올바르지 않아요. 화면에서 키를 입력해주세요.")
                    return

                print(f"🔍 검색: {query}")
                docs   = search_docs(query)
                answer = generate_answer(query, docs, claude_key)

                result = {
                    "answer": answer,
                    "sources": [
                        {
                            "filename": d["filename"],
                            "filepath": d["filepath"],
                            "preview":  d["preview"][:300],
                            "score":    d["score"],
                            "title":    d["title"],
                            "date":     d["date"],
                            "source":   d["source"],
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
    init()
    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    print(f"{'='*50}")
    print(f"  🚀 RAG 서버 실행 중")
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
