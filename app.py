"""
RAG Document Chatbot (Google Gemini + LangChain + ChromaDB)
-------------------------------------------------------------
Upload PDF / TXT / DOCX documents, store their embeddings in a local
ChromaDB vector store, and chat with your documents using Google's
Gemini models via LangChain.

Features:
- Streaming answers
- Source citations under each answer
- Conversation memory (follow-up questions work)
- Suggested example questions after upload
- Clear chat button
- Per-file delete (no need to wipe the whole DB)
- Friendly error handling for API/file issues
- Simple password authentication
- Adjustable chunk size / overlap (with document-type presets)
- Warning about ephemeral storage on Streamlit Cloud free tier

Setup:
1. pip install -r requirements.txt
2. Create a .env file with: GOOGLE_API_KEY=your_key_here
   Optionally add: APP_PASSWORD=your_password_here
3. Run: streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")  # optional - if unset, auth is skipped

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")
st.title("📄 Chat with your Documents")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ---------------------------------------------------------------------
# 4. Simple authentication (skipped automatically if APP_PASSWORD not set)
# ---------------------------------------------------------------------
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("🔒 Login required")
        pwd = st.text_input("Enter password", type="password")
        if st.button("Login"):
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

MAX_HISTORY_TURNS = 4  # how many previous Q&A pairs to feed back as context

# Document-type presets for chunk size tuning (5)
CHUNK_PRESETS = {
    "General": (1000, 150),
    "Technical / Code-heavy": (600, 100),
    "Narrative / Story": (1500, 250),
    "Legal / Dense text": (1800, 300),
    "Custom": None,
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_document(uploaded_file):
    """Save the uploaded file to a temp path and load it with the right loader."""
    suffix = "." + uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    for d in docs:
        d.metadata["source"] = uploaded_file.name

    return docs


def add_documents_to_store(uploaded_files, chunk_size, chunk_overlap):
    """
    3. Error handling: wraps loading/embedding so API errors (rate limits,
    oversized files, bad formats) show a friendly message instead of a
    raw traceback.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    total_chunks = 0
    failed_files = []

    for f in uploaded_files:
        try:
            docs = load_document(f)
            chunks = splitter.split_documents(docs)
            if not chunks:
                failed_files.append((f.name, "No readable text found in this file."))
                continue

            vectordb.add_documents(chunks)
            total_chunks += len(chunks)

        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "quota" in msg or "429" in msg:
                friendly = "API rate limit reached. Wait a moment and try again."
            elif "size" in msg or "too large" in msg or "413" in msg:
                friendly = "File is too large to process."
            elif "unsupported file type" in msg:
                friendly = "Unsupported file type."
            else:
                friendly = f"Could not process this file ({e.__class__.__name__})."
            failed_files.append((f.name, friendly))

    if total_chunks:
        vectordb.persist()

    return total_chunks, failed_files


def get_stored_sources():
    """Return a sorted list of unique source filenames currently in the DB."""
    try:
        data = vectordb._collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in data.get("metadatas", []) if m}
        return sorted(sources)
    except Exception:
        return []


def delete_source(filename):
    """2. File management: delete only the chunks belonging to one file."""
    try:
        vectordb._collection.delete(where={"source": filename})
        vectordb.persist()
        return True, None
    except Exception as e:
        return False, str(e)


def extract_text(content) -> str:
    """Normalize Gemini response content (str or list-of-blocks) into plain text."""
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()
    return content


def build_history_text() -> str:
    """Turn recent chat turns into a short text block for follow-up context."""
    history = st.session_state.messages[-(MAX_HISTORY_TURNS * 2):]
    if not history:
        return ""
    lines = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def stream_answer(query: str):
    """
    Retrieve relevant chunks, stream the LLM's answer, and yield source list.
    3. Error handling: catches retrieval/generation failures gracefully.
    """
    try:
        results = vectordb.similarity_search(query, k=4)
    except Exception:
        yield "⚠️ Couldn't search the documents right now. Please try again in a moment.", []
        return

    if not results:
        yield "I couldn't find anything relevant in the uploaded documents.", []
        return

    context_text = "\n\n---\n\n".join(
        f"(Source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for doc in results
    )

    history_text = build_history_text()
    history_block = f"Conversation so far:\n{history_text}\n\n" if history_text else ""

    prompt = (
        "You are a helpful assistant answering questions about the user's "
        "uploaded documents. Use the conversation history to understand "
        "follow-up questions (e.g. 'what about his skills?'), but answer "
        "using ONLY the document context below. If the answer isn't in the "
        "context, say you don't know based on the uploaded documents.\n\n"
        f"{history_block}"
        f"Document context:\n{context_text}\n\nQuestion: {query}"
    )

    sources = sorted({doc.metadata.get("source", "unknown") for doc in results})

    full_text = ""
    try:
        for chunk in llm.stream(prompt):
            piece = extract_text(chunk.content)
            if piece:
                full_text += piece
                yield full_text, sources
    except Exception as e:
        msg = str(e).lower()
        if "rate limit" in msg or "quota" in msg or "429" in msg:
            friendly = "⚠️ API rate limit reached. Please wait a moment and try again."
        else:
            friendly = "⚠️ Something went wrong while generating the answer. Please try again."
        if not full_text:
            yield friendly, sources
        else:
            yield full_text + f"\n\n{friendly}", sources


# ---------------------------------------------------------------------
# Sidebar: Upload documents
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Upload Documents")

    # 1. Persistent storage warning
    st.divider()

    # 5. Chunk size tuning
    st.subheader("⚙️ Chunk Settings")
    preset_choice = st.selectbox("Document type", list(CHUNK_PRESETS.keys()))

    if CHUNK_PRESETS[preset_choice] is not None:
        chunk_size, chunk_overlap = CHUNK_PRESETS[preset_choice]
        st.caption(f"Using chunk size={chunk_size}, overlap={chunk_overlap}")
    else:
        chunk_size = st.slider("Chunk size", 200, 3000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=25)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            total_chunks, failed_files = add_documents_to_store(
                uploaded_files, chunk_size, chunk_overlap
            )
            if total_chunks:
                st.success(f"Added {total_chunks} chunks from {len(uploaded_files) - len(failed_files)} file(s).")
                st.session_state["just_processed"] = True
            if failed_files:
                for fname, reason in failed_files:
                    st.error(f"❌ {fname}: {reason}")

    st.divider()

    # 2. File management - list stored files with individual delete buttons
    st.subheader("📚 Stored Files")
    sources = get_stored_sources()
    if not sources:
        st.caption("No files stored yet.")
    else:
        for src in sources:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {src}")
            if col2.button("🗑️", key=f"del_{src}", help=f"Delete {src}"):
                ok, err = delete_source(src)
                if ok:
                    st.success(f"Deleted {src}")
                    st.rerun()
                else:
                    st.error(f"Could not delete {src}: {err}")

    st.divider()
    try:
        doc_count = vectordb._collection.count()
    except Exception:
        doc_count = 0
    st.caption(f"📊 {doc_count} chunks currently stored.")

    st.divider()
    if st.button("🗑️ Clear All Chat & Files"):
        try:
            for src in get_stored_sources():
                delete_source(src)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

    if st.button("🧹 Clear Chat Only"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📎 Sources: " + ", ".join(msg["sources"]))

# Suggested example questions after a fresh upload, before the first message
if st.session_state.get("just_processed") and not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(3)
    suggestions = [
        "Summarize this document",
        "What are the key skills or highlights?",
        "What is the most important detail here?",
    ]
    suggestion_clicked = None
    for col, text in zip(cols, suggestions):
        if col.button(text):
            suggestion_clicked = text
else:
    suggestion_clicked = None

user_query = st.chat_input("Ask a question about your documents...") or suggestion_clicked

if user_query:
    st.session_state["just_processed"] = False
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        has_docs = vectordb._collection.count() > 0
    except Exception:
        has_docs = False

    with st.chat_message("assistant"):
        if not has_docs:
            answer = "Please upload and process at least one document first."
            st.markdown(answer)
            sources = []
        else:
            placeholder = st.empty()
            answer = ""
            sources = []
            for partial_answer, partial_sources in stream_answer(user_query):
                answer = partial_answer
                sources = partial_sources
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
            if sources:
                st.caption("📎 Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
