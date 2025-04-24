import streamlit as st
st.set_page_config(page_title="📚 History RAG Chatbot", layout="wide")

import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# === Environment ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyDetjWLtpDucAF1ZfaN3xLBr-wC-sqVWKc"

# === Load Bot Function ===
@st.cache_resource
def load_bot():
    doc = fitz.open("data/grade-11-history-text-book.pdf")
    pages = [(f"Page {i+1}", page.get_text()) for i, page in enumerate(doc) if page.get_text().strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for pid, text in pages:
        for chunk in splitter.split_text(text):
            documents.append(Document(page_content=chunk, metadata={"page": pid}))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return retriever, qa_chain, llm

retriever, qa_chain, fallback_llm = load_bot()

# === UI Styling ===
st.markdown("""
<style>
.big-font { font-size:25px !important; font-weight:600; }
.user-bubble {
    background-color: #dfefff;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: inline-block;
}
.ai-bubble {
    background-color: #f0f0f0;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-font'>📜 Grade 11 History AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask any question from the textbook — and if it's not there, I'll use general knowledge 🌐")

query = st.text_input("🔎 **Enter your history question**", placeholder="e.g., Who directed the Manhattan Project?")

if st.button("💬 Get Answer") and query:
    with st.spinner("🤔 Thinking with Gemini..."):
        context_docs = retriever.get_relevant_documents(query)
        context = "\n---\n".join([doc.page_content for doc in context_docs])
        pages = ", ".join(sorted({doc.metadata["page"] for doc in context_docs}))

        rag_answer = qa_chain.run(query)
        if "i don't know" in rag_answer.lower() or not rag_answer.strip():
            answer = fallback_llm.invoke(query).content
            source = "🌐 *Based on general knowledge (fallback mode).*"
        else:
            answer = rag_answer
            source = "📖 *Based on textbook content.*"

    # User question bubble
    st.markdown(f"🧑‍🎓 <div class='user-bubble'>{query}</div>", unsafe_allow_html=True)

    # AI answer bubble
    st.markdown("🤖 <div class='ai-bubble'>" + answer + "</div>", unsafe_allow_html=True)
    st.caption(source)

    if context_docs:
        with st.expander("📄 Source Pages"):
            st.write(pages)
        with st.expander("📚 Context Snippets"):
            for i, doc in enumerate(context_docs):
                st.markdown(f"**Chunk {i+1} ({doc.metadata['page']})**")
                st.write(doc.page_content)
