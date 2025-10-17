import streamlit as st
from chat_rag import build_index_from_pdf, load_index, answer_query
import os

st.title("Local PDF RAG Chatbot")

# Initialize state
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "idx" not in st.session_state:
    st.session_state.idx = None

# Upload PDF
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    pdf_path = os.path.join(os.getcwd(), uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state["pdf_path"] = pdf_path

    if st.button("Build Index"):
        with st.spinner("Building Index..."):
            idx = build_index_from_pdf(pdf_path)
            idx.save("faiss.index", "meta.pkl")
            st.session_state.idx = idx
            st.session_state.index_loaded = True
            st.success("Index built and saved!")

# Load existing index
if not st.session_state.index_loaded and os.path.exists("faiss.index") and os.path.exists("meta.pkl"):
    if st.button("Load existing index"):
        st.session_state.idx = load_index("faiss.index", "meta.pkl")
        st.session_state.index_loaded = True
        st.success("Index loaded!")

# Show question input if index is loaded
if st.session_state.index_loaded and st.session_state.idx is not None:
    if "history" not in st.session_state:
        st.session_state.history = []

        # Display previous Q&A
    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    query = st.text_input("Ask a question about the PDF", key="query")
    ask = st.button("Ask")

    if ask and query:
        with st.spinner("Retrieving answer..."):
            ans = answer_query(st.session_state.idx, query)
            st.session_state.history.append((query, ans))  # Save to memory
        st.rerun()
else:
    st.info("Please upload a PDF and build or load the index first.")