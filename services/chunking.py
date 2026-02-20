from typing import List
import tiktoken # type: ignore


def chunk_text(
    text: str,
    max_tokens: int = 700,
    overlap_paragraphs: int = 1
) -> List[str]:

    encoding = tiktoken.get_encoding("cl100k_base")

    # Better paragraph detection
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:

        para_tokens = len(encoding.encode(para))

        # If single paragraph too large → split safely
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            para_encoded = encoding.encode(para)
            for i in range(0, len(para_encoded), max_tokens):
                chunk_tokens = para_encoded[i:i+max_tokens]
                chunks.append(encoding.decode(chunk_tokens))
            continue

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))

            # PARAGRAPH OVERLAP
            overlap = current_chunk[-overlap_paragraphs:]
            current_chunk = overlap.copy()
            current_tokens = sum(
                len(encoding.encode(p)) for p in current_chunk
            )

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
