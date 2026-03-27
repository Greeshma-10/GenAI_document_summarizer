"""
Vector Store Service — Pinecone
"""

import hashlib
import re
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaLLM

from v1.services.bedrock_service import get_embedding
from v2.config import settings
from v2.logging_config import get_logger
from v2.prompts.rag import build_rag_prompt

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Replace constants block completely

UPSERT_BATCH_SIZE = settings.UPSERT_BATCH_SIZE
METADATA_CHAR_LIMIT = settings.METADATA_CHAR_LIMIT
METADATA_CHAR_MIN_FALLBACK = settings.METADATA_CHAR_MIN_FALLBACK
MIN_CHUNK_TEXT_LEN = settings.MIN_CHUNK_TEXT_LEN
MAX_SENTENCES_PER_CHUNK = settings.MAX_SENTENCES_PER_CHUNK
MIN_SENTENCE_LEN = settings.MIN_SENTENCE_LEN
MATH_SYMBOLS = set(settings.MATH_SYMBOLS)


# ─────────────────────────────────────────────────────────────────────────────
def _clean_chunk_text(text: str) -> str:
    """
    Clean a raw PDF chunk and split into readable sentences.
    Returns newline-separated sentences for bullet display,
    or empty string if result is too short.
    """
    # Fix broken hyphenated words across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Remove citation numbers
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

    clean_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if sum(1 for c in stripped if c in MATH_SYMBOLS) >= 2:
            continue
        if re.match(r'^\d+\s*$', stripped):
            continue
        if re.match(r'^(Figure|Table|Fig\.)\s*\d+', stripped):
            continue
        if len(stripped) < 15:
            continue
        clean_lines.append(stripped)

    text = ' '.join(clean_lines)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\[\s*\]', '', text).strip()

    if len(text) < MIN_CHUNK_TEXT_LEN:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > MIN_SENTENCE_LEN]

    return '\n'.join(sentences[:MAX_SENTENCES_PER_CHUNK])


# ─────────────────────────────────────────────────────────────────────────────
class VectorStore:

    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        api_key    = api_key    or settings.PINECONE_API_KEY
        index_name = index_name or settings.PINECONE_INDEX

        if not api_key:
            raise ValueError("Pinecone API key must be provided or set via PINECONE_API_KEY")

        self.pc         = Pinecone(api_key=api_key)
        self.index_name = index_name

        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            logger.info("Creating Pinecone index '%s'", index_name)
            self.pc.create_index(
                name=index_name,
                dimension=settings.PINECONE_DIMENSION,
                metric=settings.PINECONE_METRIC,
                spec=ServerlessSpec(cloud="aws", region=settings.AWS_REGION),
            )
            logger.info("Pinecone index '%s' created", index_name)
        else:
            logger.info("Pinecone index '%s' is ready", index_name)

        self.index = self.pc.Index(index_name)
        self._llm  = None   # lazy-loaded only when answer() is called

    # ─────────────────────────────────────────────────────────────────────────
    def _get_llm(self) -> OllamaLLM:
        """Lazy-load LLM — only initialised when answer() is first called."""
        if self._llm is None:
            self._llm = OllamaLLM(model="llama3")
            logger.info("RAG LLM initialised")
        return self._llm

    # ─────────────────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: List[str], doc_id: str = "default") -> int:
        """Embed and upsert chunks into Pinecone. Returns count indexed."""
        if not chunks:
            logger.warning("index_chunks called with empty chunk list | doc_id='%s'", doc_id)
            return 0

        vectors = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                clean_chunk = " ".join(chunk.split())

                if len(clean_chunk) > METADATA_CHAR_LIMIT:
                    truncated   = clean_chunk[:METADATA_CHAR_LIMIT]
                    last_period = max(
                        truncated.rfind(". "),
                        truncated.rfind("! "),
                        truncated.rfind("? "),
                    )
                    clean_chunk = (
                        truncated[:last_period + 1]
                        if last_period > METADATA_CHAR_MIN_FALLBACK
                        else truncated
                    )

                embedding = get_embedding(chunk)
                vectors.append({
                    "id":     f"{doc_id}_chunk_{i}",
                    "values": embedding,
                    "metadata": {
                        "doc_id":    doc_id,
                        "chunk_idx": i,
                        "text":      clean_chunk,
                    },
                })
            except Exception:
                logger.exception("Failed to embed chunk %d | doc_id='%s'", i, doc_id)
                continue

        if not vectors:
            logger.warning("No valid vectors produced | doc_id='%s'", doc_id)
            return 0

        for batch_start in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch     = vectors[batch_start:batch_start + UPSERT_BATCH_SIZE]
            batch_num = batch_start // UPSERT_BATCH_SIZE + 1
            self.index.upsert(vectors=batch)
            logger.info("Upserted %d chunks (batch %d) | doc_id='%s'",
                        len(batch), batch_num, doc_id)

        logger.info("Indexed %d chunks total | doc_id='%s'", len(vectors), doc_id)
        return len(vectors)

    # ─────────────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5,
               doc_id: Optional[str] = None) -> List[str]:
        """
        Semantic search — return most relevant chunks for a query.
        Retrieves top_k * 3 candidates then filters by threshold,
        so low-scoring chunks don't reduce result count.
        """
        if not query.strip():
            logger.warning("search called with empty query")
            return []

        try:
            query_embedding = get_embedding(query)
        except Exception:
            logger.exception("Failed to embed query | query='%s'", query[:80])
            return []

        try:
            filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else None

            # Retrieve 3x candidates to ensure enough pass threshold
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 3,
                include_metadata=True,
                filter=filter_dict,
            )

            all_matches = results.get("matches", [])

            # Log all scores for debugging
            scores = [round(m.get("score", 0), 3) for m in all_matches[:10]]
            logger.info("Pinecone raw scores (top 10): %s | query='%s'",
                        scores, query[:60])

            # Filter by threshold then cap at top_k
            chunks = []
            for match in all_matches:
                score = match.get("score", 0)
                text  = match.get("metadata", {}).get("text", "")
                if score > settings.SIMILARITY_THRESHOLD and text:
                    cleaned = _clean_chunk_text(text)
                    if cleaned:
                        chunks.append(cleaned)
                if len(chunks) >= top_k:
                    break

            logger.info("Search returned %d chunks (threshold=%.2f) | query='%s'",
                        len(chunks), settings.SIMILARITY_THRESHOLD, query[:60])
            return chunks

        except Exception:
            logger.exception("Pinecone search failed | query='%s'", query[:80])
            return []

    # ─────────────────────────────────────────────────────────────────────────
    def answer(self, question: str, top_k: int = 8,
               doc_id: Optional[str] = None) -> dict:
        """
        RAG: retrieve relevant chunks then generate a focused LLM answer.

        Returns:
            {
              "question":     str,
              "answer":       str,   <- LLM-generated, grounded in chunks
              "chunks":       list,  <- retrieved chunks for transparency
              "source_count": int
            }
        """
        logger.info("RAG answer | question='%s'", question[:80])

        # Step 1: retrieve relevant chunks
        chunks = self.search(question, top_k=top_k, doc_id=doc_id)

        if not chunks:
            logger.warning("No relevant chunks found | question='%s'", question[:80])
            return {
                "question":     question,
                "answer":       "No relevant information found in the document.",
                "chunks":       [],
                "source_count": 0,
            }

        # Step 2: build numbered context from chunks
        context = "\n\n".join(
            f"[Source {i+1}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        )

        # Step 3: generate focused answer using RAG prompt
        prompt = build_rag_prompt(question, context)
        try:
            llm    = self._get_llm()
            answer = str(llm.invoke(prompt)).strip()
            logger.info("RAG answer generated | sources=%d", len(chunks))
        except Exception:
            logger.exception("RAG LLM call failed | question='%s'", question[:80])
            answer = "Could not generate an answer — LLM call failed."

        return {
            "question":     question,
            "answer":       answer,
            "chunks":       chunks,
            "source_count": len(chunks),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def describe_index(self) -> dict:
        """Return index stats — useful for debugging chunk count."""
        try:
            stats = self.index.describe_index_stats()
            logger.info("Index stats: %s", stats)
            return dict(stats)
        except Exception:
            logger.exception("Failed to describe index")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        try:
            self.index.delete(filter={"doc_id": {"$eq": doc_id}})
            logger.info("Deleted all chunks | doc_id='%s'", doc_id)
        except Exception as e:
            if "Namespace not found" in str(e) or "404" in str(e):
                logger.info("No existing chunks to delete | doc_id='%s'", doc_id)
            else:
                logger.exception("Failed to delete chunks | doc_id='%s'", doc_id)

    # ─────────────────────────────────────────────────────────────────────────
    def delete_all(self) -> None:
        """Clear the entire Pinecone index."""
        try:
            self.index.delete(delete_all=True)
            logger.info("Pinecone index '%s' cleared", self.index_name)
        except Exception:
            logger.exception("Failed to clear Pinecone index '%s'", self.index_name)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def make_doc_id(filename: str) -> str:
        """Generate a stable 12-character doc_id from a filename."""
        return hashlib.md5(filename.encode()).hexdigest()[:12]