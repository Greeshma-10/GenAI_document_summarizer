from typing import List, Dict
import numpy as np # type: ignore
from sklearn.cluster import AgglomerativeClustering # type: ignore
from config import settings
from logger import logger
from services.bedrock_service import get_embedding

BASE_DISTANCE_RESEARCH = settings.BASE_DISTANCE_RESEARCH
BASE_DISTANCE_ACADEMIC = settings.BASE_DISTANCE_ACADEMIC
MIN_CHUNKS_FOR_CLUSTERING = settings.MIN_CHUNKS_FOR_CLUSTERING

# MODE-AWARE CHUNK VALIDATOR

def is_strong_chunk(chunk: Dict, mode: str = "academic") -> bool:

    summary = chunk.get("summary", "").strip()
    key_points = chunk.get("key_points", [])
    risks = chunk.get("key_risks_action_items", [])

    combined_text = " ".join(
        [summary] +
        [kp for kp in key_points if isinstance(kp, str)] +
        [r for r in risks if isinstance(r, str)]
    ).strip()

    word_count = len(combined_text.split())

    valid_key_points = [
        kp for kp in key_points
        if isinstance(kp, str) and kp.strip()
    ]

    if mode == "research":
        if word_count < 20:
            return False
        if len(valid_key_points) < 1:
            return False

    elif mode == "academic":
        if word_count < 12:
            return False

    return True

# BUILD EMBEDDINGS

def build_chunk_embeddings(chunk_summaries: List[Dict]):

    embeddings = []

    for chunk in chunk_summaries:

        text = f"""
        Summary:
        {chunk.get('summary','')}

        Key Points:
        {' '.join(chunk.get('key_points', []))}

        Risks:
        {' '.join(chunk.get('key_risks_action_items', []))}
        """

        emb = get_embedding(text)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    return embeddings

# SEMANTIC SECTION BUILDER 

def build_semantic_sections(
        chunk_summaries: List[Dict],
        mode: str = "academic",
        distance_threshold: float = None
):

    if not chunk_summaries:
        return []

    if distance_threshold is None:
        if mode == "research":
            distance_threshold = BASE_DISTANCE_RESEARCH
        else:
            distance_threshold = BASE_DISTANCE_ACADEMIC

    strong_chunks = [
        c for c in chunk_summaries
        if is_strong_chunk(c, mode=mode)
    ]

    if not strong_chunks:
        logger.warning("No semantically strong chunks found.")
        return []

    if len(strong_chunks) < MIN_CHUNKS_FOR_CLUSTERING:
        return [{
            "section_id": 0,
            "section_chunks": strong_chunks,
            "covered_chunk_ids": [c["chunk_id"] for c in strong_chunks]
        }]

    # Build embeddings

    embeddings = build_chunk_embeddings(strong_chunks)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(embeddings)

    sections = {}

    for label, chunk in zip(labels, strong_chunks):
        sections.setdefault(label, []).append(chunk)

    # Order sections by earliest chunk appearance
    ordered_sections = sorted(
        sections.items(),
        key=lambda x: min(c["chunk_id"] for c in x[1])
    )

    semantic_sections = []

    for new_id, (_, chunks) in enumerate(ordered_sections):
        semantic_sections.append({
            "section_id": new_id,
            "section_chunks": chunks,
            "covered_chunk_ids": [c["chunk_id"] for c in chunks]
        })

    num_sections = len(semantic_sections)
    total_chunks = len(strong_chunks)

    if num_sections == 1 and total_chunks > 6:
        logger.warning("Only 1 section formed from many chunks. Decreasing threshold.")
        new_threshold = max(0.05, distance_threshold - 0.05)
        if new_threshold != distance_threshold:
            return build_semantic_sections(
                strong_chunks,
                mode=mode,
                distance_threshold=new_threshold
            )
        
    if num_sections > total_chunks * 0.7:
        logger.warning(f"{num_sections} sections formed from {total_chunks} chunks. Increasing threshold.")
        new_threshold = min(0.9, distance_threshold + 0.05)
        if new_threshold != distance_threshold:
            return build_semantic_sections(
                strong_chunks,
                mode=mode,
                distance_threshold=new_threshold
            )

    return semantic_sections
