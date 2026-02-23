# GenAI Hierarchical Document Summarization Platform

> A multi-layer GenAI pipeline for summarizing long-form documents in a structured, reliable, and measurable manner.

---

## Overview

Traditional summarization systems often flatten information, omit important sections, or generate ungrounded abstractions. This platform addresses those limitations through:

- **Hierarchical abstraction**
- **Semantic clustering**
- **Evaluation-driven governance mechanisms**

The system is engineered with modular architecture, centralized configuration, structured logging, and prompt isolation to support maintainability and controlled experimentation.

---

## Solution Strategy

The platform follows a hierarchical summarization design:

1. Segment the document into token-aware chunks
2. Generate grounded chunk summaries
3. Reconstruct semantic structure using embeddings
4. Produce section-level abstractions
5. Generate a mode-aware executive synthesis
6. Validate structural and semantic completeness

Each stage introduces controlled abstraction while maintaining traceability to source content.

---

## System Architecture

The pipeline is organized into clearly separated layers.

### Interface Layer
- User uploads document (PDF or text)
- Frontend validates request and initiates pipeline

### Layer 1 — Ingestion & Preprocessing
- Token-aware chunk generation
- Detection of tabular structures
- Metadata assignment for traceability

### Layer 2 — Chunk Abstraction
- Mode-aware filtering (Research vs Academic)
- Numeric preservation logic
- Phrase and keyword grounding checks
- Parallel chunk processing for scalability

### Layer 3 — Semantic Section Reconstruction
- Embedding generation (Titan embeddings)
- Agglomerative clustering using cosine distance
- Adaptive threshold tuning to prevent cluster collapse
- Filtering of weak semantic segments

#### Research Basis for Hierarchical Clustering Strategy

The semantic section reconstruction layer is inspired by recent research demonstrating the effectiveness of hierarchical agglomerative clustering in structured summarization pipelines.

In particular, the paper:

**Dalal et al. (2024)** — *Text summarization for pharmaceutical sciences using hierarchical clustering with a weighted evaluation methodology*  
Published in *Scientific Reports (Nature)*.

The authors demonstrate that incorporating cosine-based hierarchical agglomerative clustering prior to abstractive summarization significantly improves:

- ROUGE scores  
- Semantic coherence  
- Business entity retention  
- Structural completeness  

Their results show that clustering semantically related sentences before summary generation leads to measurable improvements compared to direct summarization without clustering.

This platform extends that strategy by:
- Applying clustering to LLM-generated chunk summaries  
- Using adaptive distance thresholds instead of fixed cluster counts  
- Introducing mode-aware filtering (Research vs Academic)  
- Integrating structural coverage and semantic alignment validation  

Reference:  
Dalal, A., Ranjan, S., Bopaiah, Y., et al. (2024). *Text summarization for pharmaceutical sciences using hierarchical clustering with a weighted evaluation methodology.* Scientific Reports, 14, 20149.  
https://doi.org/10.1038/s41598-024-70618-w

### Layer 4 — Section Abstraction
- Deduplication of cross-chunk themes
- Consolidation of related insights
- Handling of sparse or low-density sections
- Preservation of comparative and tabular insights

### Layer 5 — Executive Synthesis
- Section importance ranking
- Prioritized synthesis of dominant themes
- Dual-mode prompting:
  - **Research mode** — strategic abstraction
  - **Academic mode** — explanatory abstraction
- Strict fabrication prevention constraints

### Layer 6 — Evaluation & Governance

| Engine | Description |
|---|---|
| **Structural Coverage Engine** | Measures how many meaningful chunks are represented in higher abstractions |
| **Semantic Alignment Engine** | Computes embedding similarity between section summaries and executive output to quantify meaning retention |
| **Reliability Flags** | Detects missing sections, validates coverage thresholds, and flags semantic emptiness |

### Output Layer

Structured JSON response including:

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
## Architecture diagram
![Architecture diagram](https://github.com/user-attachments/assets/a02774a3-bc2f-48b0-9394-51a283f3118c)
---

## Key Engineering Principles

### 1. Configuration-Driven Design
All runtime parameters are centralized in `config.py`, enabling safe experimentation without modifying business logic.

### 2. Prompt Isolation
Prompts are separated by stage to prevent tight coupling between orchestration and LLM policy:

```
prompts/
├── chunk.py
├── section.py
└── executive.py
```

### 3. Centralized Model Invocation
All Bedrock calls are routed through a single service layer, ensuring retry management, latency control, consistent parsing, and observability.

### 4. Structured Logging
The logging layer tracks model retries, groundedness failures, embedding issues, cluster adjustments, and executive generation failures. Logs are persisted in `logs/app.log`.

### 5. Measurable Completeness
Unlike most summarization systems, this platform introduces quantitative signals:

| Metric | Description |
|---|---|
| **Structural Coverage Score** | Chunk representation completeness |
| **Meaning Coverage Score** | Embedding similarity across abstraction levels |
| **Reliability Flags** | Structural and semantic validation signals |

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

# Running the Application

## 1. Prerequisites

- Python 3.9+
- AWS credentials configured (via `.env` or environment variables)
- Required dependencies installed

```bash
pip install -r requirements.txt
```

> Ensure your `.env` file contains required AWS and model configuration variables.

---

## 2. Start the Backend (FastAPI via Uvicorn)

The backend exposes the summarization pipeline as an API service.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Default URL:** http://localhost:8000
- **API Docs (if enabled):** http://localhost:8000/docs

---

## 3. Start the Frontend (Streamlit)

In a separate terminal:

```bash
streamlit run frontend.py
```

- **Default URL:** http://localhost:8501

The frontend connects to the running backend service and allows users to upload documents and retrieve structured summaries.

---

## 4. End-to-End Flow

```
Start backend (Uvicorn)
        ↓
Start frontend (Streamlit)
        ↓
Upload document via UI
        ↓
Pipeline executes through hierarchical layers
        ↓
Structured JSON response is rendered in the interface
```

## Performance Characteristics

- Parallel chunk processing
- Stage-level latency measurement
- End-to-end runtime tracking

Empirical tests across increasing token sizes demonstrate **stable structural coverage** and **strong semantic alignment** across abstraction levels.

---

## Future Enhancements

- Retrieval-augmented grounding verification
- Multimodal document support
- Model benchmarking framework

---

## Summary

This project demonstrates how long-form document summarization can be approached as a **structured, hierarchical, and evaluable system** rather than a single-pass generation task. It integrates abstraction control, semantic reconstruction, grounding validation, and measurable completeness into a cohesive GenAI architecture.
