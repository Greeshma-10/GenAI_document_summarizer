import streamlit as st # type: ignore
import requests # type: ignore
import json
import time

API_URL = "http://127.0.0.1:8000/summarize"

st.set_page_config(
    page_title="GenAI Document Summarizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR 

with st.sidebar:
    st.title("⚙️ Controls")

    mode = st.selectbox(
        "Document Mode",
        ["academic", "research"],
        help="""
Academic → Relaxed grounding, suitable for textbooks/tutorials  
Research → Strict grounding, suitable for research papers
"""
    )

    st.markdown("---")

    coverage_container = st.container()

# MAIN HEADER

st.title("📄 GenAI Document Summarization Pipeline")
st.caption("Hierarchical summarization with coverage validation and responsible AI checks.")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

# PROCESS

if uploaded_file is not None:

    if st.button("Generate Summary", use_container_width=True):

        progress = st.progress(0)

        with st.spinner("Processing document..."):
            start = time.time()

            if uploaded_file.name.endswith(".pdf"):
                mime_type = "application/pdf"
            elif uploaded_file.name.endswith(".txt"):
                mime_type = "text/plain"
            else:
                st.error("Unsupported file type.")
                st.stop()

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    mime_type
                )
            }

            data = {"mode": mode}

            try:
                response = requests.post(API_URL, files=files, data=data)
            except Exception as e:
                st.error(f"Backend connection failed: {str(e)}")
                st.stop()

            end = time.time()

        progress.progress(100)

        if response.status_code == 200:

            result = response.json()
            doc = result.get("document_summary", {})
            performance = result.get("performance", {})

            st.success(f"Completed in {round(end - start, 2)} seconds")
            st.info(f"Mode Used: {mode.upper()}")

            # COVERAGE (SIDEBAR)

            with coverage_container:
                st.markdown("## 📊 Coverage")

                coverage = doc.get("coverage_score", 0)
                meaning = doc.get("meaning_coverage_score", 0)

                st.metric("Coverage Score", f"{coverage}%")
                st.progress(min(int(coverage), 100))

                st.metric("Meaning Coverage", f"{meaning}%")
                st.progress(min(int(meaning), 100))

                st.metric("Missing Section Flag", doc.get("missing_section_flag", False))

            st.divider()

            # EXECUTIVE SUMMARY

            st.header("📌 Executive Summary")

            st.subheader("TL;DR")
            st.write(doc.get("tldr", ""))

            st.subheader("Executive Summary")
            st.write(doc.get("executive_summary", ""))

            st.subheader("Key Points")
            for point in doc.get("key_points", []):
                st.markdown(f"- {point}")

            st.subheader("Risks / Action Items")
            for risk in doc.get("risks_action_items", []):
                st.markdown(f"- {risk}")

            st.divider()

            # SECTION SUMMARIES

            st.header("📂 Section Summaries")

            for section in doc.get("sections", []):

                with st.expander(f"Section {section['section_id']}"):

                    st.write(section.get("section_summary", ""))

                    st.markdown("**Key Points:**")
                    for kp in section.get("section_key_points", []):
                        st.markdown(f"- {kp}")

                    st.markdown("**Risks / Action Items:**")
                    for r in section.get("section_risks_action_items", []):
                        st.markdown(f"- {r}")

                    st.markdown("**Covered Chunks:**")
                    st.write(section.get("covered_chunk_ids", []))

            st.divider()

            # PERFORMANCE DASHBOARD

            with st.expander("⚙️ Performance Metrics (Advanced)"):

                col1, col2, col3 = st.columns(3)

                col1.metric("Ingestion", f"{performance.get('ingestion_time_sec', 0)}s")
                col2.metric("Chunking", f"{performance.get('chunking_time_sec', 0)}s")
                col3.metric("Chunk Summarization", f"{performance.get('chunk_summarization_time_sec', 0)}s")

                col4, col5, col6 = st.columns(3)

                col4.metric("Section Build", f"{performance.get('section_build_time_sec', 0)}s")
                col5.metric("Section Summarization", f"{performance.get('section_summarization_time_sec', 0)}s")
                col6.metric("Executive", f"{performance.get('executive_time_sec', 0)}s")

                st.metric("Total Time", f"{performance.get('total_time_sec', 0)}s")

                st.json(performance)

            # DOWNLOAD OUTPUT

            st.download_button(
                label="⬇ Download Summary JSON",
                data=json.dumps(result, indent=2),
                file_name="summary_output.json",
                mime="application/json",
                use_container_width=True
            )

        else:
            st.error(f"Backend error: {response.status_code}")
