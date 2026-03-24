"""
v2/prompts/rag.py

RAG (Retrieval Augmented Generation) prompt.
Used by VectorStore.answer() to generate a focused answer
from retrieved chunks.
"""


def build_rag_prompt(question: str, context: str) -> str:
    return f"""You are a helpful research assistant answering questions about a document.

Answer the question using ONLY the context provided below.
- Be concise and specific
- Use bullet points if listing multiple facts
- If the context does not contain enough information to answer, say "The document does not contain enough information to answer this question."
- Do NOT make up information not present in the context

Context:
\"\"\"{context}\"\"\"

Question: {question}

Answer:"""