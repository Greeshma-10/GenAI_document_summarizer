# GenAI Hierarchical Document Summarization Platform

> A multi-layer GenAI pipeline for summarizing long-form documents in a structured, reliable, and measurable manner.

---

## Overview

Traditional summarization systems often flatten information, omit important sections, or generate ungrounded abstractions. This platform addresses those limitations through **hierarchical abstraction**, **semantic clustering**, and **evaluation-driven governance mechanisms**.

The system is engineered with modular architecture, centralized configuration, structured logging, and prompt isolation to support maintainability and controlled experimentation.

---

## Solution Strategy

The pipeline follows a hierarchical summarization design:

1. Segment the document into token-aware chunks
2. Generate grounded chunk summaries
3. Reconstruct semantic structure using embeddings
4. Produce section-level abstractions
5. Generate a mode-aware executive synthesis
6. Validate structural and semantic completeness

---

## System Architecture

![Architecture Diagram](https://github.com/user-attachments/assets/a02774a3-bc2f-48b0-9394-51a283f3118c)

### Interface Layer
- User uploads document (PDF or text)
- Frontend validates request and initiates pipeline

### Layer 1 — Ingestion & Preprocessing
- Token-aware chunk generation with metadata assignment
- Detection of tabular structures

### Layer 2 — Chunk Abstraction
- Mode-aware filtering (Research vs Academic)
- Numeric preservation and phrase grounding checks
- Parallel chunk processing for scalability

### Layer 3 — Semantic Section Reconstruction
- Embedding generation via Titan Embeddings
- Agglomerative clustering using cosine distance
- Adaptive threshold tuning to prevent cluster collapse

> **Research Basis:** This layer is inspired by Dalal et al. (2024), which demonstrated that cosine-based hierarchical agglomerative clustering prior to abstractive summarization significantly improves ROUGE scores, semantic coherence, and structural completeness. ([DOI: 10.1038/s41598-024-70618-w](https://doi.org/10.1038/s41598-024-70618-w))

### Layer 4 — Section Abstraction
- Deduplication of cross-chunk themes
- Handling of sparse sections and preservation of tabular insights

### Layer 5 — Executive Synthesis
- Section importance ranking with prioritized synthesis
- **Research mode** — strategic abstraction
- **Academic mode** — explanatory abstraction
- Strict fabrication prevention constraints

### Layer 6 — Evaluation & Governance

| Engine | Description |
|---|---|
| **Structural Coverage Engine** | Measures chunk representation in higher abstractions |
| **Semantic Alignment Engine** | Computes embedding similarity between section and executive output |
| **Reliability Flags** | Detects missing sections and flags semantic emptiness |

### Output

```json
{
  "executive_summary": "...",
  "key_points": ["..."],
  "risks_or_limitations": ["..."],
  "tldr": "...",
  "coverage_metrics": {},
  "reliability_flags": {}
}
```

---

## Key Engineering Principles

- **Configuration-Driven Design** — All runtime parameters are centralized in `config.py`
- **Prompt Isolation** — Prompts are separated by stage (`chunk.py`, `section.py`, `executive.py`)
- **Centralized Model Invocation** — All Bedrock calls route through a single service layer for retry, latency, and observability
- **Structured Logging** — Tracks retries, failures, and cluster adjustments; persisted in `logs/app.log`
- **Measurable Completeness** — Structural coverage score, meaning coverage score, and reliability flags

---

## Repository Structure

```
├── app/
├── prompts/
│   ├── chunk.py
│   ├── section.py
│   └── executive.py
├── services/
├── schema/
├── logs/
│   └── app.log
├── config.py
├── logger.py
└── frontend.py
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Cloud AI | AWS Bedrock |
| LLM | LLaMA 3 |
| Embeddings | Titan Embeddings |
| Clustering | Scikit-learn |
| Numerics | NumPy |
| Frontend | Streamlit |

---

## Running the Application

### Prerequisites

- Python 3.9+
- AWS credentials configured (via `.env` or environment variables)

```bash
pip install -r requirements.txt
```

### Start the Backend (FastAPI via Uvicorn)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Default URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Start the Frontend (Streamlit)

```bash
streamlit run frontend.py
```

- **Default URL:** http://localhost:8501

### End-to-End Flow

```
Start backend → Start frontend → Upload document → Pipeline runs → JSON response rendered
```

---

## Performance & Future Enhancements

Empirical tests show stable structural coverage and strong semantic alignment across abstraction levels, with parallel chunk processing and stage-level latency tracking.

**Planned:**
- Retrieval-augmented grounding verification
- Multimodal document support
- Model benchmarking framework
