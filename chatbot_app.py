import streamlit as st
st.set_page_config(page_title="History Chatbot", layout="wide")

import os
import json
import re
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import FakeEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

# Load environment
os.environ["GOOGLE_API_KEY"] = "AIzaSyDetjWLtpDucAF1ZfaN3xLBr-wC-sqVWKc"

def extract_section_title(text):
    lines = text.split("\n")
    for line in lines:
        if re.match(r'^(Chapter|SECTION|CHAPTER|Activity|Lesson|Unit|British|Dutch|Portuguese|Kandyan)', line.strip(), re.IGNORECASE):
            return line.strip()
    return "General"

@st.cache_resource
def load_bot():
    doc = fitz.open("data/grade-11-history-text-book.pdf")
    pages = [(f"Page {i+1}", page.get_text()) for i, page in enumerate(doc) if page.get_text().strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for pid, text in pages:
        section = extract_section_title(text)
        for chunk in splitter.split_text(text):
            documents.append(Document(page_content=chunk, metadata={"page": pid, "section": section}))

    embeddings = FakeEmbeddings(size=1536)
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return retriever, qa_chain

retriever, qa_chain = load_bot()

# === Streamlit UI ===
st.title("📚 Grade 11 History AI Chatbot")
st.markdown("Ask anything from the Grade 11 textbook 📖")

query = st.text_input("Enter your question:", placeholder="Why did the English begin to focus on Sri Lanka in the 18th century?")

if st.button("💬 Get Answer") and query:
    with st.spinner("Thinking with Gemini 🤖..."):
        context_docs = retriever.get_relevant_documents(query)
        filtered_docs = [doc for doc in context_docs if any(kw in doc.page_content.lower() for kw in ["british", "sri lanka", "18th century", "colonial", "rule", "english"])]
        context = "\n---\n".join([doc.page_content for doc in filtered_docs])
        pages = ", ".join(sorted({doc.metadata["page"] for doc in filtered_docs}))
        answer = qa_chain.run(query) if context else "Sorry, this textbook may not contain a detailed answer to your question."

    st.subheader("🔍 Answer")
    st.write(answer)

    with st.expander("📄 Source Pages"):
        st.write(pages)

    with st.expander("📚 Context Snippets"):
        for i, doc in enumerate(filtered_docs):
            st.markdown(f"**Chunk {i+1} ({doc.metadata['page']})**")
            st.write(doc.page_content)
