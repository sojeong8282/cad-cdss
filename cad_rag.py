"""cad_rag.py — 진료지침 기반 RAG(검색증강생성) 모듈.

CAD CDSS 챗봇/SOAP가 LLM의 머릿속 지식이 아니라, 미리 넣어둔 진료지침 문서에서
질문과 관련된 부분을 '검색'해 근거로 함께 사용하도록 만든다(환각 감소 + 출처 제시).

설계 원칙
- 외부 벡터DB(FAISS/Chroma 등) 없이 numpy 코사인 유사도로 검색한다.
  (KB가 수십~수백 청크 규모라 충분하고, 설치 마찰이 없다.)
- 임베딩 백엔드:
    1순위 OpenAI(text-embedding-3-small)  — OPENAI_API_KEY가 있을 때
    폴백   TF-IDF(scikit-learn, 문자 n-gram) — 키가 없어도 오프라인 동작
- guidelines/ 폴더의 .md/.txt 문서를 청크로 나눠 색인한다.

이 모듈만 단독으로 import/테스트할 수 있도록 streamlit에 의존하지 않는다.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

DEFAULT_GUIDELINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "guidelines"
)


# -----------------------------------------------------------------------------
# 문서 로드 & 청크 분할
# -----------------------------------------------------------------------------
def _split_into_chunks(text: str, source: str, max_chars: int = 600) -> list[dict]:
    """문단 단위로 자르고, 너무 길면 문장 단위로 추가 분할한다."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: list[str] = []
    for block in blocks:
        # 마크다운 제목 줄은 청크 본문에서 제외(맥락만 약하게 유지)
        block = re.sub(r"^#+\s.*$", "", block, flags=re.MULTILINE).strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        sentences = re.split(r"(?<=[.!?。])\s+", block)
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) > max_chars:
                chunks.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            chunks.append(current.strip())
    return [{"text": c, "source": source} for c in chunks]


def load_documents(guideline_dir: str = DEFAULT_GUIDELINE_DIR) -> list[dict]:
    """guidelines 폴더의 모든 .md/.txt를 읽어 청크 리스트로 반환한다."""
    paths = sorted(
        glob.glob(os.path.join(guideline_dir, "*.md"))
        + glob.glob(os.path.join(guideline_dir, "*.txt"))
    )
    chunks: list[dict] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        title = os.path.splitext(os.path.basename(path))[0]
        chunks.extend(_split_into_chunks(text, source=title))
    return chunks


# -----------------------------------------------------------------------------
# 임베딩 백엔드
# -----------------------------------------------------------------------------
class _OpenAIEmbedder:
    name = "OpenAI(text-embedding-3-small)"

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # 지연 import

        self.client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype="float32")


class _TfidfEmbedder:
    name = "TF-IDF(scikit-learn, 오프라인 폴백)"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 문자 n-gram: 한국어 형태소 분석 없이도 부분 일치를 잡아준다
        # (예: '가슴이 조인다' ↔ '흉통/압박' 같은 표현 차이에 어느 정도 견고)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1
        )

    def fit(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray().astype("float32")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype("float32")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# -----------------------------------------------------------------------------
# 검색기
# -----------------------------------------------------------------------------
class GuidelineRetriever:
    """진료지침 청크를 색인하고, 질의와 유사한 청크를 검색한다."""

    def __init__(
        self,
        guideline_dir: str = DEFAULT_GUIDELINE_DIR,
        prefer_openai: bool = True,
    ):
        self.chunks = load_documents(guideline_dir)
        self.texts = [c["text"] for c in self.chunks]
        self.matrix: np.ndarray | None = None
        self.backend: str | None = None
        self._mode: str | None = None

        if not self.texts:
            return

        embedder = None
        if prefer_openai and os.getenv("OPENAI_API_KEY"):
            try:
                embedder = _OpenAIEmbedder()
                self.matrix = _l2_normalize(embedder.embed(self.texts))
                self.backend = embedder.name
                self._embedder = embedder
                self._mode = "openai"
            except Exception:
                embedder = None  # 실패 시 폴백

        if embedder is None:
            tfidf = _TfidfEmbedder()
            self.matrix = _l2_normalize(tfidf.fit(self.texts))
            self.backend = tfidf.name
            self._tfidf = tfidf
            self._mode = "tfidf"

    @property
    def ready(self) -> bool:
        return self.matrix is not None and len(self.chunks) > 0

    @property
    def num_docs(self) -> int:
        return len({c["source"] for c in self.chunks})

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._mode == "openai":
            return _l2_normalize(self._embedder.embed([query]))[0]
        return _l2_normalize(self._tfidf.embed([query]))[0]

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
        """질의와 가장 유사한 청크 상위 k개를 반환한다."""
        if not self.ready or not (query or "").strip():
            return []
        q_vec = self._embed_query(query)
        scores = self.matrix @ q_vec
        order = np.argsort(scores)[::-1][:k]
        results = []
        for i in order:
            score = float(scores[i])
            if score < min_score:
                continue
            results.append(
                {
                    "text": self.chunks[i]["text"],
                    "source": self.chunks[i]["source"],
                    "score": round(score, 3),
                }
            )
        return results


def build_grounding_block(results: list[dict]) -> str:
    """검색 결과를 LLM 프롬프트에 넣을 근거 블록 문자열로 만든다."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[근거 {i}] (출처: {r['source']})\n{r['text']}")
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# (보너스) 표준 용어 의미 매칭 — TERM_MAP의 정확 일치 한계를 보완
# -----------------------------------------------------------------------------
def semantic_match_term(phrase: str, term_to_synonyms: dict[str, list[str]],
                        min_score: float = 0.2) -> tuple[str | None, float]:
    """자유 표현(phrase)을 표준 용어 사전에 의미 유사도로 매칭한다.

    예: '가슴이 조이는 느낌' → 'Typical Chest Pain'.
    외부 의존 없이 문자 n-gram TF-IDF 코사인으로 동작(오프라인).
    반환: (가장 가까운 표준용어 또는 None, 점수)
    """
    phrase = (phrase or "").strip()
    if not phrase or not term_to_synonyms:
        return None, 0.0
    from sklearn.feature_extraction.text import TfidfVectorizer

    labels, docs = [], []
    for term, synonyms in term_to_synonyms.items():
        bag = " ".join([term] + list(synonyms))
        labels.append(term)
        docs.append(bag)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    mat = _l2_normalize(vec.fit_transform(docs).toarray().astype("float32"))
    q = _l2_normalize(vec.transform([phrase]).toarray().astype("float32"))[0]
    scores = mat @ q
    best = int(np.argmax(scores))
    best_score = float(scores[best])
    if best_score < min_score:
        return None, best_score
    return labels[best], round(best_score, 3)


if __name__ == "__main__":
    # 간단 자체 점검
    r = GuidelineRetriever()
    print(f"backend={r.backend} | docs={r.num_docs} | chunks={r.num_chunks}")
    for q in ["운동할 때 가슴이 조이는데 위험한가요?", "EF가 낮으면 무슨 의미죠?", "추가로 어떤 검사를 하나요?"]:
        print(f"\nQ: {q}")
        for hit in r.retrieve(q, k=2):
            print(f"  [{hit['score']}] ({hit['source']}) {hit['text'][:60]}...")
