import streamlit as st
import pandas as pd

from evaluation.run_eval import evaluate_iter  # ✅ exists after you add it

st.set_page_config(page_title="RAG Evaluation", layout="wide")

st.title("📊 RAG Evaluation Dashboard")
st.caption("Runs evaluation using evaluation/tests.jsonl and your implementation.answer pipeline.")

run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
st.divider()

if run_btn:
    progress = st.progress(0)
    status = st.empty()

    rows = []
    for i, total, test, row in evaluate_iter():
        rows.append(row)
        progress.progress(i / total)
        status.write(f"Running {i}/{total} — {test.category} — {test.question}")

    status.success("✅ Evaluation complete")

    df = pd.DataFrame(rows)

    # Summary
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg keyword recall", f"{df['keyword_recall'].mean():.3f}")
    with c2:
        st.metric("Avg answer/ref cosine", f"{df['answer_ref_cosine'].mean():.3f}")
    with c3:
        st.metric("Retrieval pass rate", f"{(df['passed_retrieval'].mean()*100):.1f}%")

    st.divider()

    # By category
    st.subheader("By category")
    cat = (
        df.groupby("category", as_index=False)
        .agg(avg_keyword_recall=("keyword_recall", "mean"),
             avg_answer_ref_cosine=("answer_ref_cosine", "mean"),
             pass_rate=("passed_retrieval", "mean"),
             n=("category", "count"))
        .sort_values("avg_keyword_recall", ascending=False)
    )
    cat["pass_rate"] = cat["pass_rate"] * 100.0

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.caption("Avg keyword recall")
        st.bar_chart(cat.set_index("category")["avg_keyword_recall"])
    with col2:
        st.caption("Avg answer/ref cosine")
        st.bar_chart(cat.set_index("category")["avg_answer_ref_cosine"])

    st.dataframe(cat, use_container_width=True)

    st.divider()

    # Full results
    st.subheader("All results")
    df_show = df.copy()
    df_show["keyword_hits"] = df_show["keyword_hits"].apply(
        lambda d: ", ".join([k for k, v in d.items() if v])
    )
    st.dataframe(df_show, use_container_width=True, height=550)
