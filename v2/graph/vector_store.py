"""
Vector Store Service — Pinecone

Handles:
  1. index_chunks()   — embed and store document chunks in Pinecone
  2. search()         — semantic search for relevant chunks given a query
  3. delete_all()     — clear the index for a new document
"""

import os
import hashlib
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from v1.services.bedrock_service import get_embedding


class VectorStore:

    def __init__(self):
        api_key    = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX", "docmind")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")

        self.pc         = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Create index if it doesn't exist
        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            print(f"📌 Creating Pinecone index '{index_name}'...")
            self.pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{index_name}' created")
        else:
            print(f"✅ Pinecone index '{index_name}' ready")

        self.index = self.pc.Index(index_name)

    # ─────────────────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: List[str], doc_id: str = "default") -> int:
        """
        Embed and upsert chunks into Pinecone.
        Each chunk gets a unique ID based on doc_id + position.

        Returns number of chunks indexed.
        """
        if not chunks:
            return 0

        vectors = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                # Clean chunk — remove excessive whitespace, normalize line breaks
                clean_chunk = " ".join(chunk.split())

                # Pinecone metadata limit is 40KB — cap at 3000 chars to be safe
                # but keep whole sentences
                if len(clean_chunk) > 3000:
                    # truncate at last sentence boundary before 3000 chars
                    truncated = clean_chunk[:3000]
                    last_period = max(
                        truncated.rfind(". "),
                        truncated.rfind("! "),
                        truncated.rfind("? ")
                    )
                    if last_period > 500:
                        clean_chunk = truncated[:last_period + 1]
                    else:
                        clean_chunk = truncated

                embedding = get_embedding(chunk)   # embed original, store cleaned
                chunk_id  = f"{doc_id}_chunk_{i}"
                vectors.append({
                    "id":     chunk_id,
                    "values": embedding,
                    "metadata": {
                        "doc_id":    doc_id,
                        "chunk_idx": i,
                        "text":      clean_chunk
                    }
                })
            except Exception as e:
                print(f"⚠️ Failed to embed chunk {i}: {e}")
                continue

        if not vectors:
            return 0

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"📥 Pinecone: upserted {len(batch)} chunks (batch {i//batch_size + 1})")

        print(f"✅ Indexed {len(vectors)} chunks for doc_id='{doc_id}'")
        return len(vectors)

    # ─────────────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[str]:
        """
        Semantic search — find the most relevant chunks for a query.

        Args:
            query:  the search query (e.g. a claim to verify)
            top_k:  number of results to return
            doc_id: if provided, filter to chunks from this document only

        Returns:
            List of chunk text strings, most relevant first
        """
        if not query.strip():
            return []

        try:
            query_embedding = get_embedding(query)
        except Exception as e:
            print(f"⚠️ Failed to embed query: {e}")
            return []

        try:
            filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else None

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            chunks = []
            for match in results.get("matches", []):
                score = match.get("score", 0)
                text  = match.get("metadata", {}).get("text", "")
                if score > 0.4 and text:   # return reasonably similar chunks
                    chunks.append(text)

            print(f"🔍 Pinecone search: {len(chunks)} relevant chunks "
                  f"(score > 0.5) for query: '{query[:60]}...'")
            return chunks

        except Exception as e:
            print(f"⚠️ Pinecone search failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    def delete_document(self, doc_id: str):
        """Delete all chunks for a specific document."""
        try:
            self.index.delete(filter={"doc_id": {"$eq": doc_id}})
            print(f"🗑️ Deleted all chunks for doc_id='{doc_id}'")
        except Exception as e:
            # 404 just means index is empty — not an error
            if "Namespace not found" in str(e) or "404" in str(e):
                print(f"ℹ️ No existing chunks for doc_id='{doc_id}' (index empty)")
            else:
                print(f"⚠️ Failed to delete chunks for '{doc_id}': {e}")

    # ─────────────────────────────────────────────────────────────────────────
    def delete_all(self):
        """Clear the entire index."""
        try:
            self.index.delete(delete_all=True)
            print("🗑️ Pinecone index cleared")
        except Exception as e:
            print(f"⚠️ Failed to clear index: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def make_doc_id(filename: str) -> str:
        """Generate a stable doc_id from a filename."""
        return hashlib.md5(filename.encode()).hexdigest()[:12]