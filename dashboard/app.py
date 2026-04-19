import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Benchmark Harness",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 RAG Benchmark Harness")
st.caption("India Union Budget 2026-27 | LLMOps Reference Implementation")

tab1, tab2, tab3 = st.tabs([
    "💬 Live Query",
    "📊 Experiment Results",
    "🔍 Model Comparison"
])


# ── Tab 1: Live Query ─────────────────────────────────────────────
with tab1:
    st.header("Ask a question about Budget 2026-27")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Configuration")
        model = st.selectbox(
            "Generator Model",
            ["llama-3.1-8b", "llama-3.3-70b", "claude-haiku"],
            index=0
        )
        retrieval = st.selectbox(
            "Retrieval Method",
            ["hybrid", "dense", "bm25"],
            index=0
        )
        reranking = st.checkbox("Enable Reranking", value=True)
        top_k = st.slider("Top-K chunks", 3, 10, 5)
        prompt_v = st.selectbox(
            "Prompt Version",
            ["1.0", "1.1", "1.2"],
            index=0
        )

    with col1:
        question = st.text_area(
            "Your question",
            value="What is the fiscal deficit target for 2026-27?",
            height=100
        )

        if st.button("🚀 Ask", type="primary"):
            with st.spinner("Retrieving and generating..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": question,
                            "model_key": model,
                            "retrieval_method": retrieval,
                            "reranking": reranking,
                            "top_k": top_k,
                            "prompt_version": prompt_v
                        },
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()

                        st.success("Answer")
                        st.write(data["answer"])

                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Latency", f"{data['latency_ms']:.0f}ms")
                        col_b.metric("Tokens", data["total_tokens"])
                        col_c.metric("Cost", f"${data['cost_usd']:.6f}")
                        col_d.metric("Model", data["model_key"])

                        if data.get("langfuse_trace_id"):
                            st.info(
                                f"🔍 LangFuse Trace: `{data['langfuse_trace_id']}` "
                                f"— [View on LangFuse](https://cloud.langfuse.com)"
                            )

                        with st.expander("Top Retrieved Chunk"):
                            st.text(data["top_chunk_preview"])
                    else:
                        st.error(f"API error: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to API. "
                        "Make sure FastAPI is running: `uvicorn api.main:app --port 8000`"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Tab 2: Experiment Results ─────────────────────────────────────
with tab2:
    st.header("Experiment Results")

    if st.button("🔄 Refresh Results"):
        st.cache_data.clear()

    try:
        response = requests.get(f"{API_URL}/experiments", timeout=10)

        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])

            if not experiments:
                st.warning(
                    "No experiments found. "
                    "Run: `python run_all_experiments.py`"
                )
            else:
                df = pd.DataFrame(experiments)

                # Display table
                st.subheader(f"All Runs ({len(df)} experiments)")

                display_cols = [
                    "experiment_name", "model_key", "retrieval_method",
                    "chunk_size", "reranking", "prompt_version",
                    "deepeval_faithfulness", "deepeval_answerrelevancy",
                    "deepeval_contextualprecision", "deepeval_contextualrecall",
                    "ragas_context_recall", "ragas_answer_correctness"
                ]
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)

                # Faithfulness chart
                if "deepeval_faithfulness" in df.columns:
                    st.subheader("Faithfulness by Experiment")
                    fig = px.bar(
                        df.dropna(subset=["deepeval_faithfulness"]),
                        x="experiment_name",
                        y="deepeval_faithfulness",
                        color="model_key",
                        title="DeepEval Faithfulness Score",
                        labels={
                            "deepeval_faithfulness": "Faithfulness",
                            "experiment_name": "Experiment"
                        }
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                # Radar chart for best config
                st.subheader("Metric Comparison")
                best = df.iloc[0]
                categories = [
                    "Faithfulness", "Answer Relevancy",
                    "Contextual Precision", "Contextual Recall"
                ]
                values = [
                    best.get("deepeval_faithfulness", 0) or 0,
                    best.get("deepeval_answerrelevancy", 0) or 0,
                    best.get("deepeval_contextualprecision", 0) or 0,
                    best.get("deepeval_contextualrecall", 0) or 0,
                ]

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=best.get("experiment_name", "Best Run")
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Best Config — Metric Radar"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.error(f"API error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running.")
    except Exception as e:
        st.error(f"Error: {e}")


# ── Tab 3: Model Comparison ───────────────────────────────────────
with tab3:
    st.header("Model Comparison — OSS vs Proprietary")

    try:
        response = requests.get(f"{API_URL}/experiments", timeout=10)

        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])

            if not experiments:
                st.warning("Run experiments first to see model comparison.")
            else:
                df = pd.DataFrame(experiments)

                # Filter model comparison experiments
                model_exps = df[df["experiment_name"].isin([
                    "exp_01_baseline",
                    "exp_07_llama_70b",
                    "exp_08_claude_haiku"
                ])]

                if not model_exps.empty and "deepeval_faithfulness" in df.columns:
                    st.subheader("Quality vs Cost")

                    fig = go.Figure()

                    for _, row in model_exps.iterrows():
                        fig.add_trace(go.Bar(
                            name=row["experiment_name"],
                            x=["Faithfulness", "Answer Relevancy",
                               "Contextual Precision", "Contextual Recall"],
                            y=[
                                row.get("deepeval_faithfulness", 0) or 0,
                                row.get("deepeval_answerrelevancy", 0) or 0,
                                row.get("deepeval_contextualprecision", 0) or 0,
                                row.get("deepeval_contextualrecall", 0) or 0,
                            ]
                        ))

                    fig.update_layout(
                        barmode="group",
                        title="Model Quality Comparison (DeepEval)",
                        yaxis=dict(range=[0, 1]),
                        legend_title="Experiment"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Key Finding")
                    st.info(
                        "💡 **Retrieval strategy matters more than model choice.** "
                        "Moving from dense-only to hybrid + reranking "
                        "gave larger faithfulness improvements than "
                        "switching from Llama 8B to Claude Haiku."
                    )

                # LangFuse link
                st.subheader("Observability")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        "**LangFuse** — Full trace viewer\n\n"
                        "[Open LangFuse Dashboard](https://cloud.langfuse.com)"
                    )
                with col2:
                    st.markdown(
                        "**LangSmith** — Parallel trace viewer\n\n"
                        "[Open LangSmith Dashboard](https://smith.langchain.com)"
                    )

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
    except Exception as e:
        st.error(f"Error: {e}")


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **RAG Benchmark Harness**

    Systematic benchmarking of RAG configurations
    on India Union Budget 2026-27.

    **Stack:**
    - Retrieval: ChromaDB + BM25 + Hybrid
    - Reranking: Cross-Encoder
    - LLMs: Groq (OSS) + Claude (Proprietary)
    - Observability: LangFuse + LangSmith
    - Evaluation: DeepEval + Ragas
    - Tracking: MLflow

    **Research Question:**
    Which RAG configuration gives the best
    quality-to-cost ratio?
    """)

    st.header("API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline")
