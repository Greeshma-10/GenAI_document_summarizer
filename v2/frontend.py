import streamlit as st
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocMind — Knowledge Extraction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f5f0e8; color: #1a1a2e; }
section[data-testid="stSidebar"] { background-color: #1a1a2e; border-right: none; }
section[data-testid="stSidebar"] * { color: #e8e0d0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p { color: #a09880 !important; }
.stButton > button {
    background-color: #1a1a2e; color: #f5f0e8; border: none; border-radius: 6px;
    font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 500;
    padding: 0.6rem 1.2rem; transition: all 0.2s ease; letter-spacing: 0.02em;
}
.stButton > button:hover { background-color: #2d2d4e; color: #f5c842; transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] {
    background-color: #ede8e0; border-radius: 8px; padding: 4px;
    gap: 2px; border: 1px solid #d4cfc4;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #6b6560;
    font-family: 'DM Mono', monospace; font-size: 0.78rem; border-radius: 6px; padding: 6px 16px;
}
.stTabs [aria-selected="true"] { background-color: #1a1a2e !important; color: #f5c842 !important; }
.main-title {
    font-family: 'DM Serif Display', serif; font-size: 3.2rem; color: #1a1a2e;
    line-height: 1; letter-spacing: -0.02em; margin-bottom: 0;
}
.main-title span { color: #c8742a; font-style: italic; }
.sub-caption {
    font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #8a8070;
    margin-top: 0.4rem; letter-spacing: 0.05em;
}
.step-pill {
    display: inline-block; background: #1a1a2e; color: #f5c842;
    padding: 3px 14px; border-radius: 4px; font-size: 0.7rem;
    font-family: 'DM Mono', monospace; letter-spacing: 0.08em; margin-bottom: 1.2rem;
}
.card {
    background: #ffffff; border: 1px solid #e0d8cc; border-radius: 10px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(26,26,46,0.06);
}
.card-amber {
    background: #fffbf0; border: 1px solid #e8d48a; border-left: 4px solid #f5c842;
    border-radius: 0 10px 10px 0; padding: 1rem 1.4rem; margin-bottom: 1rem;
}
.card-warning {
    background: #fff8ee; border: 1px solid #f5c842; border-left: 4px solid #c8742a;
    border-radius: 0 10px 10px 0; padding: 1rem 1.4rem; margin-bottom: 1rem;
    font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #7a4010;
}
.badge-supported {
    background: #e8f5ee; color: #1a6b3c; border: 1px solid #b8ddc8;
    padding: 3px 12px; border-radius: 4px; font-size: 0.72rem;
    font-family: 'DM Mono', monospace; font-weight: 500; letter-spacing: 0.05em;
}
.badge-contradicted {
    background: #fdf0ee; color: #8b2020; border: 1px solid #e8b8b8;
    padding: 3px 12px; border-radius: 4px; font-size: 0.72rem;
    font-family: 'DM Mono', monospace; font-weight: 500; letter-spacing: 0.05em;
}
.badge-unverified {
    background: #fdf8e8; color: #7a5c10; border: 1px solid #e8d48a;
    padding: 3px 12px; border-radius: 4px; font-size: 0.72rem;
    font-family: 'DM Mono', monospace; font-weight: 500; letter-spacing: 0.05em;
}
.metric-block {
    background: #ffffff; border: 1px solid #e0d8cc; border-radius: 10px;
    padding: 1.2rem; text-align: center; box-shadow: 0 1px 4px rgba(26,26,46,0.06);
}
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: #1a1a2e; line-height: 1; }
.metric-label {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #8a8070;
    letter-spacing: 0.08em; margin-top: 6px; text-transform: uppercase;
}
.entity-tag {
    display: inline-block; background: #f0ece4; border: 1px solid #d4cfc4;
    color: #3a3060; padding: 3px 10px; border-radius: 4px; font-size: 0.76rem;
    margin: 3px; font-family: 'DM Mono', monospace;
}
.triple-row {
    background: #fafaf7; border-left: 3px solid #c8742a; padding: 0.6rem 1rem;
    margin-bottom: 0.4rem; border-radius: 0 6px 6px 0;
    font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #3a3050;
    border-top: 1px solid #ede8e0; border-right: 1px solid #ede8e0; border-bottom: 1px solid #ede8e0;
}
.triple-subj { color: #1a1a2e; font-weight: 500; }
.triple-rel  { color: #c8742a; }
.triple-obj  { color: #3a3060; }
.section-label {
    font-size: 0.68rem; font-family: 'DM Mono', monospace; color: #8a8070;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.6rem;
}
.sidebar-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: #f5f0e8; }
.sidebar-sub { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #6a6458; letter-spacing: 0.05em; }
.pipeline-step { font-family: 'DM Mono', monospace; font-size: 0.78rem; margin: 5px 0; }
hr { border-color: #ddd8cc; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff; border: 1px solid #d4cfc4; border-radius: 6px;
    font-family: 'DM Sans', sans-serif; color: #1a1a2e;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #c8742a; box-shadow: 0 0 0 2px rgba(200,116,42,0.15);
}
.stSelectbox > div > div { background: #ffffff; border: 1px solid #d4cfc4; border-radius: 6px; }
.stRadio > div { gap: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key in ["summarize_result", "section_summaries",
            "executive_summary", "entities", "source_text"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "graph_built" not in st.session_state:
    st.session_state.graph_built = False

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">DocMind</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">knowledge extraction pipeline</p>', unsafe_allow_html=True)
    st.markdown("---")

    mode = st.selectbox(
        "Document Mode", ["academic", "research"],
        help="Academic: relaxed grounding\nResearch: strict grounding for papers"
    )

    st.markdown("---")
    st.markdown('<p class="section-label" style="color:#6a6458">Pipeline Status</p>',
                unsafe_allow_html=True)
    for label, done in {
        "01 · Summarize":  st.session_state.summarize_result is not None,
        "02 · Graph Build": st.session_state.graph_built,
    }.items():
        icon  = "✓" if done else "○"
        color = "#f5c842" if done else "#4a4860"
        st.markdown(
            f'<p class="pipeline-step" style="color:{color}">{icon}  {label}</p>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.session_state.summarize_result:
        doc     = st.session_state.summarize_result.get("document_summary", {})
        meaning = doc.get("meaning_coverage_score", 0)
        st.markdown('<p class="section-label" style="color:#6a6458">Meaning Coverage</p>',
                    unsafe_allow_html=True)
        st.metric("", f"{meaning}%")
        st.progress(min(int(meaning), 100))

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">Doc<span>Mind</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-caption">MULTI-MODAL SUMMARIZATION · ENTITY EXTRACTION · '
    'KNOWLEDGE GRAPH · FACT VERIFICATION</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄  Summarize", "🕸  Graph", "🔍  Query", "✅  Fact Check", "📊  Evaluate",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="step-pill">STEP 01 · UPLOAD & SUMMARIZE</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"],
                                      label_visibility="collapsed")

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            run_summarize = st.button("⚡  Generate Summary", use_container_width=True)
        with col2:
            run_graph = st.button("🕸  Build Knowledge Graph", use_container_width=True)

        mime = "application/pdf" if uploaded_file.name.endswith(".pdf") else "text/plain"

        # ── SUMMARIZE ─────────────────────────────────────────────────────────
        if run_summarize:
            with st.spinner("Processing document..."):
                uploaded_file.seek(0)
                try:
                    resp = requests.post(
                        f"{BASE_URL}/summarize",
                        files={"file": (uploaded_file.name, uploaded_file, mime)},
                        data={"mode": mode},
                        timeout=600                         # 10 min for large docs
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.summarize_result = result
                        doc = result.get("document_summary", {})
                        st.session_state.section_summaries = doc.get("sections", [])
                        st.session_state.executive_summary = doc.get("executive_summary", "")
                        st.session_state.entities = result.get("entities", {})
                        st.success("Summary generated!")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out — the document may be too large. Try a shorter PDF.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        # ── BUILD GRAPH ───────────────────────────────────────────────────────
        if run_graph:
            # ← warn user before starting so they don't click away
            st.markdown(
                '<div class="card-warning">'
                '⏳ &nbsp;Graph build can take <strong>5–15 minutes</strong> for a full paper. '
                'Do not close this tab or refresh the page.'
                '</div>',
                unsafe_allow_html=True
            )
            with st.spinner("Extracting entities and building graph — please wait..."):
                uploaded_file.seek(0)
                try:
                    resp = requests.post(
                        f"{BASE_URL}/graph/build",
                        files={"file": (uploaded_file.name, uploaded_file, mime)},
                        data={"mode": mode},
                        timeout=1800                        # ← 30 min, was 600
                    )
                    if resp.status_code == 200:
                        r = resp.json()
                        st.session_state.graph_built = True
                        st.success(
                            f"Graph built — {r.get('num_relations', 0)} triples, "
                            f"{r.get('num_entities', 0)} entities"
                        )
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except requests.exceptions.Timeout:
                    st.error(
                        "Graph build timed out. Try setting MAX_CHUNKS = 8 in graph_pipeline.py "
                        "to process fewer chunks and finish faster."
                    )
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.summarize_result:
        result = st.session_state.summarize_result
        doc    = result.get("document_summary", {})
        perf   = result.get("performance", {})

        st.markdown("---")
        st.markdown("### Executive Summary")
        st.markdown(
            f'<div class="card-amber"><strong>TL;DR</strong><br>{doc.get("tldr","")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="card">{doc.get("executive_summary","")}</div>',
                    unsafe_allow_html=True)

        col_kp, col_ri = st.columns(2)
        with col_kp:
            st.markdown("**Key Points**")
            for kp in doc.get("key_points", []):
                st.markdown(f"- {kp}")
        with col_ri:
            st.markdown("**Risks / Action Items**")
            for r in doc.get("risks_action_items", []):
                st.markdown(f"- {r}")

        st.markdown("---")
        st.markdown("### Extracted Entities")
        entities = result.get("entities", {})
        if entities:
            for cat, items in entities.items():
                if items:
                    st.markdown(
                        f'<p class="section-label">{cat.replace("_"," ").upper()}</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "".join(f'<span class="entity-tag">{i}</span>' for i in items),
                        unsafe_allow_html=True
                    )
                    st.markdown("")

        st.markdown("---")
        st.markdown("### Section Summaries")
        for sec in doc.get("sections", []):
            with st.expander(f"Section {sec.get('section_id','')}"):
                st.write(sec.get("section_summary", ""))
                for kp in sec.get("section_key_points", []):
                    st.markdown(f"- {kp}")

        st.markdown("---")
        with st.expander("⚙  Performance Metrics"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Ingestion",  f"{perf.get('ingestion_time_sec',0)}s")
            c2.metric("Chunking",   f"{perf.get('chunking_time_sec',0)}s")
            c3.metric("Chunk Summ", f"{perf.get('chunk_summarization_time_sec',0)}s")
            c4, c5, c6 = st.columns(3)
            c4.metric("Section Build", f"{perf.get('section_build_time_sec',0)}s")
            c5.metric("Section Summ",  f"{perf.get('section_summarization_time_sec',0)}s")
            c6.metric("Executive",     f"{perf.get('executive_time_sec',0)}s")
            st.metric("Total", f"{perf.get('total_time_sec',0)}s")

        st.download_button(
            "⬇  Download Summary JSON",
            data=json.dumps(result, indent=2),
            file_name="summary_output.json",
            mime="application/json",
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="step-pill">STEP 02 · KNOWLEDGE GRAPH</p>', unsafe_allow_html=True)

    if not st.session_state.graph_built:
        st.markdown(
            '<div class="card-amber">Build the graph first using the button in the Summarize tab.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="card" style="border-left:4px solid #1a6b3c">✓ &nbsp;Graph is ready in Neo4j</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Browse by Entity Type")
        entity_type = st.selectbox(
            "Type", ["MODEL","ORGANIZATION","METRIC","DATASET","TASK","CONCEPT"],
            label_visibility="collapsed"
        )
        if st.button("Browse", use_container_width=True):
            with st.spinner("Fetching..."):
                try:
                    resp = requests.get(f"{BASE_URL}/graph/query",
                                         params={"type":"by_type","entity_type":entity_type,"limit":30})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(f"**{data.get('count',0)} {entity_type} entities**")
                        for ent in data.get("entities", []):
                            with st.expander(ent["entity"]):
                                for rel in ent.get("relations", []):
                                    st.markdown(
                                        f'<div class="triple-row">'
                                        f'<span class="triple-subj">{ent["entity"]}</span>'
                                        f' <span class="triple-rel"> ─[{rel["relation"]}]→ </span>'
                                        f'<span class="triple-obj">{rel["target"]}</span>'
                                        f'</div>', unsafe_allow_html=True
                                    )
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")
        st.markdown("#### Subgraph Explorer")
        col_e, col_d = st.columns([3, 1])
        with col_e:
            sub_entity = st.text_input("Entity", placeholder="e.g. Transformer",
                                        label_visibility="collapsed")
        with col_d:
            depth = st.number_input("Depth", 1, 4, 2, label_visibility="collapsed")

        if st.button("Explore", use_container_width=True):
            with st.spinner("Fetching subgraph..."):
                try:
                    resp = requests.get(f"{BASE_URL}/graph/query",
                                         params={"type":"subgraph","entity":sub_entity,
                                                 "depth":depth,"limit":50})
                    if resp.status_code == 200:
                        data = resp.json()
                        c1, c2 = st.columns(2)
                        c1.metric("Nodes", data.get("node_count", 0))
                        c2.metric("Edges", data.get("edge_count", 0))
                        for edge in data.get("edges", []):
                            st.markdown(
                                f'<div class="triple-row">'
                                f'<span class="triple-subj">{edge["source"]}</span>'
                                f' <span class="triple-rel"> ─[{edge["relation"]}]→ </span>'
                                f'<span class="triple-obj">{edge["target"]}</span>'
                                f'</div>', unsafe_allow_html=True
                            )
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUERY
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="step-pill">STEP 03 · QUERY THE GRAPH</p>', unsafe_allow_html=True)

    qmode = st.radio("Mode",
                      ["Natural Language", "Entity Neighbours", "Path Between Entities"],
                      horizontal=True, label_visibility="collapsed")

    if qmode == "Natural Language":
        nl_q = st.text_input("Question",
                              placeholder="Which models were developed by Google Brain?",
                              label_visibility="collapsed")
        if st.button("Ask", use_container_width=True) and nl_q:
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(f"{BASE_URL}/graph/ask",
                                          json={"question": nl_q, "limit": 20})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(
                            f'<div class="card"><p class="section-label">Generated Cypher</p>'
                            f'<code style="color:#c8742a;font-family:DM Mono,monospace;font-size:0.8rem">'
                            f'{data.get("cypher","")}</code></div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**{data.get('count',0)} results**")
                        if data.get("results"):
                            st.dataframe(data["results"], use_container_width=True)
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

    elif qmode == "Entity Neighbours":
        ent_q   = st.text_input("Entity", placeholder="e.g. Transformer",
                                 label_visibility="collapsed")
        limit_q = st.slider("Max results", 5, 50, 15)
        if st.button("Search", use_container_width=True) and ent_q:
            with st.spinner("Fetching..."):
                try:
                    resp = requests.get(f"{BASE_URL}/graph/query",
                                         params={"type":"neighbours","entity":ent_q,"limit":limit_q})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(f"**{data.get('count',0)} relationships**")
                        for row in data.get("results", []):
                            st.markdown(
                                f'<div class="triple-row">'
                                f'<span class="triple-subj">{row["source"]}</span>'
                                f' <span class="triple-rel"> ─[{row["relation"]}]→ </span>'
                                f'<span class="triple-obj">{row["target"]}</span>'
                                f'</div>', unsafe_allow_html=True
                            )
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

    else:
        cf, ct = st.columns(2)
        with cf:
            from_e = st.text_input("From", placeholder="e.g. BLEU", label_visibility="collapsed")
        with ct:
            to_e   = st.text_input("To",   placeholder="e.g. Google Brain", label_visibility="collapsed")
        if st.button("Find Path", use_container_width=True) and from_e and to_e:
            with st.spinner("Finding path..."):
                try:
                    resp = requests.get(f"{BASE_URL}/graph/query",
                                         params={"type":"path","from":from_e,"to":to_e})
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("found"):
                            st.markdown(f"**Path found — {data['hops']} hops**")
                            for step in data.get("path", []):
                                st.markdown(
                                    f'<div class="triple-row">'
                                    f'<span class="triple-subj">{step["from"]}</span>'
                                    f' <span class="triple-rel"> ─[{step["relation"]}]→ </span>'
                                    f'<span class="triple-obj">{step["to"]}</span>'
                                    f'</div>', unsafe_allow_html=True
                                )
                        else:
                            st.warning("No path found between these entities.")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FACT CHECK
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="step-pill">STEP 04 · FACT VERIFICATION</p>', unsafe_allow_html=True)

    cmode = st.radio("Mode", ["Single Claim", "Batch Claims"],
                      horizontal=True, label_visibility="collapsed")

    if cmode == "Single Claim":
        claim_in = st.text_input("Claim",
                                  placeholder="The Transformer was developed by Google Brain",
                                  label_visibility="collapsed")
        if st.button("Verify", use_container_width=True) and claim_in:
            with st.spinner("Verifying..."):
                try:
                    resp = requests.post(f"{BASE_URL}/fact/verify", json={"claim": claim_in})
                    if resp.status_code == 200:
                        data    = resp.json()
                        verdict = data.get("verdict", "UNVERIFIED")
                        badge   = {"SUPPORTED":"badge-supported",
                                   "CONTRADICTED":"badge-contradicted",
                                   "UNVERIFIED":"badge-unverified"}.get(verdict,"badge-unverified")
                        st.markdown(
                            f'<div class="card"><span class="{badge}">{verdict}</span>'
                            f'<br><br><strong>Claim:</strong> {data["claim"]}'
                            f'<br><strong>Reason:</strong> {data.get("reason","")}'
                            f'<br><strong>Confidence:</strong> {data.get("confidence",0)}</div>',
                            unsafe_allow_html=True
                        )
                        if data.get("graph_evidence"):
                            st.markdown("**Graph Evidence**")
                            for row in data["graph_evidence"][:5]:
                                st.markdown(
                                    f'<div class="triple-row">'
                                    f'<span class="triple-subj">{row["source"]}</span>'
                                    f' <span class="triple-rel"> ─[{row["relation"]}]→ </span>'
                                    f'<span class="triple-obj">{row["target"]}</span>'
                                    f'</div>', unsafe_allow_html=True
                                )
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

    else:
        batch_text = st.text_area(
            "Claims (one per line)", height=150,
            placeholder="The Transformer was developed by Google Brain\n"
                        "The Transformer uses convolutional layers\n"
                        "BLEU is used to evaluate translation",
            label_visibility="collapsed"
        )
        if st.button("Verify All", use_container_width=True) and batch_text:
            claims = [c.strip() for c in batch_text.strip().split("\n") if c.strip()]
            with st.spinner(f"Verifying {len(claims)} claims..."):
                try:
                    resp = requests.post(f"{BASE_URL}/fact/verify/batch", json={"claims": claims})
                    if resp.status_code == 200:
                        data = resp.json()
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(
                            f'<div class="metric-block">'
                            f'<div class="metric-value" style="color:#1a6b3c">{data["supported"]}</div>'
                            f'<div class="metric-label">Supported</div></div>', unsafe_allow_html=True
                        )
                        c2.markdown(
                            f'<div class="metric-block">'
                            f'<div class="metric-value" style="color:#8b2020">{data["contradicted"]}</div>'
                            f'<div class="metric-label">Contradicted</div></div>', unsafe_allow_html=True
                        )
                        c3.markdown(
                            f'<div class="metric-block">'
                            f'<div class="metric-value" style="color:#7a5c10">{data["unverified"]}</div>'
                            f'<div class="metric-label">Unverified</div></div>', unsafe_allow_html=True
                        )
                        st.markdown("---")
                        for r in data.get("results", []):
                            verdict = r.get("verdict","UNVERIFIED")
                            badge = {"SUPPORTED":"badge-supported",
                                     "CONTRADICTED":"badge-contradicted",
                                     "UNVERIFIED":"badge-unverified"}.get(verdict,"badge-unverified")
                            st.markdown(
                                f'<div class="card"><span class="{badge}">{verdict}</span>'
                                f'&nbsp;&nbsp;<span style="color:#1a1a2e">{r["claim"]}</span>'
                                f'<br><span style="color:#8a8070;font-size:0.8rem">'
                                f'{r.get("reason","")}</span></div>', unsafe_allow_html=True
                            )
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="step-pill">STEP 05 · EVALUATION</p>', unsafe_allow_html=True)

    if st.session_state.summarize_result:
        doc              = st.session_state.summarize_result.get("document_summary", {})
        default_exec     = doc.get("executive_summary", "")
        default_entities = st.session_state.entities or {}
        default_sections = [
            {"section_summary": s.get("section_summary", "")}
            for s in doc.get("sections", []) if s.get("section_summary")
        ]
    else:
        default_exec, default_entities, default_sections = "", {}, []

    exec_text  = st.text_area("Executive Summary", value=default_exec, height=100,
                               help="Auto-populated from Summarize tab")
    max_claims = st.slider("Max claims to verify", 3, 20, 8)

    if st.button("Run Evaluation", use_container_width=True):
        if not default_sections and not exec_text:
            st.warning("Run Summarize first.")
        else:
            with st.spinner("Running full evaluation..."):
                payload = {
                    "section_summaries":  default_sections or [{"section_summary": exec_text}],
                    "executive_summary":  exec_text,
                    "extracted_entities": default_entities,
                    "max_claims":         max_claims,
                }
                try:
                    resp = requests.post(f"{BASE_URL}/evaluate", json=payload, timeout=600)
                    if resp.status_code == 200:
                        data = resp.json()
                        cov  = data.get("coverage", {})
                        fact = data.get("factual_accuracy", {})
                        ent  = data.get("entity_accuracy", {})

                        c1, c2, c3 = st.columns(3)
                        c1.markdown(
                            f'<div class="metric-block">'
                            f'<div class="metric-value">{cov.get("score",0)}</div>'
                            f'<div class="metric-label">Coverage Score</div></div>',
                            unsafe_allow_html=True
                        )
                        c2.markdown(
                            f'<div class="metric-block">'
                            f'<div class="metric-value">{fact.get("score",0)}</div>'
                            f'<div class="metric-label">Factual Accuracy</div></div>',
                            unsafe_allow_html=True
                        )
                        ent_score = ent.get("details",{}).get("overall",{}).get("score",0)
                        c3.markdown(
                            f'<div class="metric-block">'
                            f'<div class="metric-value">{ent_score}</div>'
                            f'<div class="metric-label">Entity Accuracy</div></div>',
                            unsafe_allow_html=True
                        )

                        st.markdown("---")
                        st.markdown("### Claim-by-Claim Breakdown")
                        for r in fact.get("details",{}).get("details",[]):
                            verdict = r.get("verdict","UNVERIFIED")
                            badge = {"SUPPORTED":"badge-supported",
                                     "CONTRADICTED":"badge-contradicted",
                                     "UNVERIFIED":"badge-unverified"}.get(verdict,"badge-unverified")
                            st.markdown(
                                f'<div class="card"><span class="{badge}">{verdict}</span>'
                                f'&nbsp;&nbsp;<span style="color:#1a1a2e;font-size:0.85rem">'
                                f'{r["claim"]}</span>'
                                f'<br><span style="color:#8a8070;font-size:0.78rem">'
                                f'{r.get("reason","")}</span></div>',
                                unsafe_allow_html=True
                            )

                        st.markdown("---")
                        st.markdown("### Entity Accuracy — Per Category")
                        per_cat    = ent.get("details",{}).get("per_category",{})
                        mode_label = ent.get("details",{}).get("mode","self_consistency")
                        st.caption(f"Mode: {mode_label}")

                        for cat, info in per_cat.items():
                            with st.expander(cat.replace("_"," ").upper()):
                                if mode_label == "reference":
                                    mc, rc, fc = st.columns(3)
                                    mc.metric("Precision", info.get("precision",0))
                                    rc.metric("Recall",    info.get("recall",0))
                                    fc.metric("F1",        info.get("f1",0))
                                else:
                                    cc, dc = st.columns(2)
                                    cc.metric("Count",      info.get("count",0))
                                    dc.metric("Dedup Rate", info.get("dedup_rate",0))
                                    if info.get("entities"):
                                        st.markdown(
                                            "".join(f'<span class="entity-tag">{e}</span>'
                                                    for e in info["entities"]),
                                            unsafe_allow_html=True
                                        )

                        st.markdown("---")
                        st.download_button(
                            "⬇  Download Evaluation JSON",
                            data=json.dumps(data, indent=2),
                            file_name="evaluation_output.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("Evaluation timed out — try reducing Max claims to verify.")
                except Exception as e:
                    st.error(str(e))