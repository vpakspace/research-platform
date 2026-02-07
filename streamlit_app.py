"""Research Intelligence Platform — Streamlit UI."""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional

import streamlit as st

# Load .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from research_platform.models import Citation, PerplexityModel, SearchResult, ResearchProject
from research_platform.perplexity_client import PerplexityClient
from research_platform.storage import ResearchStorage
from research_platform.pipeline import ResearchPipeline

# --- Page config ---
st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="🔬",
    layout="wide",
)

# --- Session state defaults ---
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None


def _key(s: str) -> str:
    """Deterministic widget key."""
    return hashlib.md5(s.encode()).hexdigest()[:10]


# --- Lazy singletons ---
@st.cache_resource
def get_perplexity_client() -> Optional[PerplexityClient]:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        return None
    return PerplexityClient(api_key=api_key)


@st.cache_resource
def get_storage() -> Optional[ResearchStorage]:
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        return None
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        storage = ResearchStorage(
            qdrant_url=qdrant_url,
            openai_api_key=openai_key,
        )
        storage.ensure_collection()
        return storage
    except Exception:
        return None


def get_pipeline() -> Optional[ResearchPipeline]:
    client = get_perplexity_client()
    storage = get_storage()
    if client and storage:
        return ResearchPipeline(perplexity=client, storage=storage)
    return None


# --- Sidebar ---
st.sidebar.title("Research Intelligence Platform")
st.sidebar.markdown("---")

perplexity_ok = bool(os.getenv("PERPLEXITY_API_KEY"))
openai_ok = bool(os.getenv("OPENAI_API_KEY"))

st.sidebar.markdown(f"Perplexity API: {'connected' if perplexity_ok else 'missing key'}")
st.sidebar.markdown(f"OpenAI Embeddings: {'connected' if openai_ok else 'missing key'}")

storage = get_storage()
if storage:
    stats = storage.get_stats()
    st.sidebar.markdown(f"Qdrant: {stats.get('points_count', 0)} results stored")
else:
    st.sidebar.markdown("Qdrant: not connected")

st.sidebar.markdown("---")
st.sidebar.markdown("Port: 8503")

# --- Tabs ---
tab_search, tab_projects, tab_knowledge, tab_history = st.tabs(
    ["Search", "Research Projects", "Knowledge Base", "History"]
)

# ===================== Tab 1: Search =====================
with tab_search:
    st.header("Web Search via Perplexity AI")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Query", placeholder="What is Retrieval-Augmented Generation?")
    with col2:
        model = st.selectbox(
            "Model",
            [m.value for m in PerplexityModel],
            index=1,  # sonar-pro
        )

    if st.button("Search", type="primary", disabled=not perplexity_ok):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            pipeline = get_pipeline()
            if pipeline is None:
                st.error("Pipeline not configured. Check API keys.")
            else:
                with st.spinner("Searching..."):
                    try:
                        result = pipeline.quick_search(query, model=model)

                        st.subheader("Answer")
                        st.markdown(result.answer)

                        if result.citations:
                            st.subheader(f"Sources ({len(result.citations)})")
                            for i, c in enumerate(result.citations, 1):
                                label = c.title or c.url
                                st.markdown(f"{i}. [{label}]({c.url})")
                                if c.snippet:
                                    st.caption(c.snippet)

                        st.caption(
                            f"Model: {result.model_used} | "
                            f"Tokens: {result.tokens_used} | "
                            f"ID: {result.id}"
                        )

                        # Save to session history
                        st.session_state.search_history.append(
                            {
                                "query": result.query,
                                "model": result.model_used,
                                "tokens": result.tokens_used,
                                "citations": len(result.citations),
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "id": result.id,
                            }
                        )

                    except Exception as e:
                        st.error(f"Search failed: {e}")

    if not perplexity_ok:
        st.info("Set PERPLEXITY_API_KEY environment variable to enable search.")

# ===================== Tab 2: Research Projects =====================
with tab_projects:
    st.header("Research Projects")

    with st.expander("Create New Project", expanded=False):
        proj_name = st.text_input("Project Name", key="proj_name")
        proj_desc = st.text_area("Description", key="proj_desc")
        proj_tags = st.text_input("Tags (comma-separated)", key="proj_tags")

        if st.button("Create Project"):
            if not proj_name.strip():
                st.warning("Enter a project name.")
            else:
                project = ResearchProject(
                    name=proj_name,
                    description=proj_desc,
                    tags=[t.strip() for t in proj_tags.split(",") if t.strip()],
                )
                st.session_state.projects[project.id] = project
                st.success(f"Project created: {project.name} (ID: {project.id[:8]})")

    if st.session_state.projects:
        st.subheader("Your Projects")

        for pid, proj in st.session_state.projects.items():
            with st.expander(f"{proj.name} ({len(proj.results)} results)"):
                st.write(f"Description: {proj.description}")
                st.write(f"Tags: {', '.join(proj.tags)}")

                questions_input = st.text_area(
                    "Questions (one per line)",
                    key=_key(f"q_{pid}"),
                    height=100,
                )

                if st.button("Run Research", key=_key(f"run_{pid}")):
                    questions = [
                        q.strip()
                        for q in questions_input.strip().split("\n")
                        if q.strip()
                    ]
                    if not questions:
                        st.warning("Add at least one question.")
                    else:
                        pipeline = get_pipeline()
                        if pipeline is None:
                            st.error("Pipeline not configured.")
                        else:
                            with st.spinner(f"Researching {len(questions)} questions..."):
                                try:
                                    results = pipeline.deep_research(
                                        proj.name, questions, project_id=pid
                                    )
                                    proj.queries.extend(questions)
                                    proj.results.extend(results)
                                    st.success(f"Completed {len(results)} queries!")

                                    for r in results:
                                        st.markdown(f"**Q:** {r.query}")
                                        st.markdown(r.answer[:500])
                                        if r.citations:
                                            st.caption(
                                                f"{len(r.citations)} sources | "
                                                f"{r.tokens_used} tokens"
                                            )
                                        st.markdown("---")
                                except Exception as e:
                                    st.error(f"Research failed: {e}")

                # Show existing results
                if proj.results:
                    st.subheader("Results")
                    for r in proj.results:
                        with st.expander(f"Q: {r.query[:60]}..."):
                            st.markdown(r.answer)
                            if r.citations:
                                for c in r.citations:
                                    label = c.title or c.url
                                    st.markdown(f"- [{label}]({c.url})")
    else:
        st.info("No projects yet. Create one above.")

# ===================== Tab 3: Knowledge Base =====================
with tab_knowledge:
    st.header("Knowledge Base — Semantic Search")

    kb_query = st.text_input("Search past results", placeholder="RAG architecture patterns")
    kb_limit = st.slider("Max results", 1, 20, 5)

    if st.button("Find Similar", disabled=storage is None):
        if not kb_query.strip():
            st.warning("Enter a search query.")
        elif storage is None:
            st.error("Storage not connected.")
        else:
            with st.spinner("Searching knowledge base..."):
                try:
                    results = storage.search_similar(kb_query, limit=kb_limit)
                    if results:
                        for i, r in enumerate(results, 1):
                            with st.expander(f"{i}. {r.query[:80]}"):
                                st.markdown(r.answer[:500])
                                st.caption(
                                    f"Model: {r.model_used} | "
                                    f"Citations: {len(r.citations)} | "
                                    f"Tokens: {r.tokens_used}"
                                )
                    else:
                        st.info("No similar results found.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    if storage is None:
        st.info("Set OPENAI_API_KEY and ensure Qdrant is running to use Knowledge Base.")

    # Stats
    if storage:
        st.subheader("Storage Stats")
        stats = storage.get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Results", stats.get("points_count", 0))
        col2.metric("Collection", stats.get("collection", ""))
        col3.metric("Status", stats.get("status", "unknown"))

# ===================== Tab 4: History =====================
with tab_history:
    st.header("Search History")

    history = st.session_state.search_history

    if history:
        import pandas as pd

        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)

        if st.button("Export to JSON"):
            json_str = json.dumps(history, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"research_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        st.caption(f"Total: {len(history)} searches")
    else:
        st.info("No search history yet. Run a search in the Search tab.")
