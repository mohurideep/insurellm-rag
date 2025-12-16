import streamlit as st
from dotenv import load_dotenv

from implementation.answer import answer_question
from implementation.ingest import fetch_documents, create_chunks, create_embeddings

load_dotenv(override=True)

st.set_page_config(page_title="Insurellm Expert Assistant", layout="wide")

def format_context_docs(docs):
    # Keep it simple + readable in Streamlit
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        doc_type = d.metadata.get("doc_type", "unknown")
        out.append(f"**Source:** `{src}`  \n**Type:** `{doc_type}`\n\n{d.page_content}")
    return "\n\n---\n\n".join(out) if out else "_No context retrieved._"

if "history" not in st.session_state:
    st.session_state.history = []  # list[dict] with role/content

st.title("🏢 Insurellm Expert Assistant")
st.caption("Ask anything about Insurellm. Left = chat. Right = retrieved context.")

top_col1, top_col2 = st.columns([1, 1])

with top_col1:
    if st.button("📥 Ingest knowledge base now", use_container_width=True):
        with st.spinner("Ingesting documents into Chroma..."):
            docs = fetch_documents()
            chunks = create_chunks(docs)
            create_embeddings(chunks)
        st.success("Ingestion complete. Your vector DB is updated.")

with top_col2:
    st.info("If retrieval looks wrong, you probably ingested with different embeddings than you query with. Set `EMBEDDINGS_PROVIDER` consistently in `.env`.")

col_chat, col_ctx = st.columns([1, 1], gap="large")

with col_chat:
    st.subheader("💬 Conversation")

    # Render message history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about Insurellm...")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prior = st.session_state.history[:-1]
                answer, context_docs = answer_question(user_input, prior)
            st.markdown(answer)

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state["last_context_docs"] = context_docs

with col_ctx:
    st.subheader("📚 Retrieved Context")
    docs = st.session_state.get("last_context_docs", [])
    st.markdown(format_context_docs(docs))
