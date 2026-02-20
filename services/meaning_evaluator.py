import numpy as np # type: ignore
from services.bedrock_service import get_embedding


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return float(
        np.dot(vec1, vec2) /
        (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )


def compute_meaning_coverage(section_summaries, executive_summary_text):

    section_text = " ".join(
        s.get("section_summary", "")
        for s in section_summaries
        if s.get("section_summary")
    )

    if not section_text.strip() or not executive_summary_text.strip():
        return 0.0

    section_embedding = get_embedding(section_text)
    executive_embedding = get_embedding(executive_summary_text)

    similarity = cosine_similarity(section_embedding, executive_embedding)

    return round(similarity * 100, 2)
