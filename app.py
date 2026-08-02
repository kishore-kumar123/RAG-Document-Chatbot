"""
RAG Document Chatbot - Streamlit Web UI
Browser la chat pannalam. Run panna: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA


DATA_DIR = "documents"
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📚", layout="centered")


def load_documents():
    all_docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return all_docs

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                all_docs.extend(loader.load())
            elif filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                all_docs.extend(loader.load())
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                all_docs.extend(loader.load())
        except Exception as e:
            st.warning(f"'{filename}' load panna mudiyala: {e}")

    return all_docs


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    return vectorstore


def load_existing_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


@st.cache_resource
def get_qa_chain():
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY .env file la illa!")
        st.stop()

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = load_existing_vectorstore()
    else:
        docs = load_documents()
        if not docs:
            st.error("Edhavadhu document illa. 'documents' folder la PDF/DOCX/TXT files podunga.")
            st.stop()
        with st.spinner("Documents process aaguthu... (first time mattum)"):
            vectorstore = build_vectorstore(docs)

    return create_qa_chain(vectorstore)


# ---- UI ----

st.title("📚 RAG Document Chatbot")
st.caption("Ungal documents-ai patthi kelvi kekkalam — Gemini + LangChain + ChromaDB")

qa_chain = get_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ungal kelvi enna?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Yosikuren..."):
            try:
                result = qa_chain.invoke({"query": query})
                answer = result["result"]

                sources = set()
                for doc in result.get("source_documents", []):
                    sources.add(os.path.basename(doc.metadata.get("source", "Unknown")))

                if sources:
                    answer += f"\n\n*Source: {', '.join(sources)}*"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions based on documents in the `documents/` folder.")
    st.write("**Tech stack:** Gemini, LangChain, ChromaDB")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
