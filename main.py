"""
RAG Document Chatbot
Ungal documents (PDF/DOCX/TXT) padichu, kelvi kekkirathukku Gemini + LangChain + ChromaDB use pannuthu.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA


DATA_DIR = "documents"
PERSIST_DIR = "chroma_db"


def load_documents():
    all_docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"'{DATA_DIR}' folder create pannirukken. Athula unga documents podunga, thirumba run pannunga.")
        return all_docs

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                all_docs.extend(loader.load())
                print(f"Loaded PDF: {filename}")
            elif filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                all_docs.extend(loader.load())
                print(f"Loaded DOCX: {filename}")
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                all_docs.extend(loader.load())
                print(f"Loaded TXT: {filename}")
        except Exception as e:
            print(f"'{filename}' load panna mudiyala: {e}")

    return all_docs


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Total {len(chunks)} chunks create aachu.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    print("Vector database create aachu (ChromaDB).")
    return vectorstore


def load_existing_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY .env file la illa! Athai add pannitu thirumba try pannunga.")
        return

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Existing vector database kandupudichen, athai load pannuren...")
        vectorstore = load_existing_vectorstore()
    else:
        print("Documents load panniten...")
        docs = load_documents()
        if not docs:
            print("Edhavadhu document illa. 'documents' folder la PDF/DOCX/TXT files podunga.")
            return
        vectorstore = build_vectorstore(docs)

    qa_chain = create_qa_chain(vectorstore)

    print("\n" + "=" * 50)
    print("RAG Document Chatbot Ready!")
    print("Kelvi kekkalam. Exit pannanumna 'exit' or 'quit' type pannunga.")
    print("=" * 50 + "\n")

    while True:
        query = input("Neenga: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break
        if not query:
            continue
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\nBot: {result['result']}\n")
            sources = set()
            for doc in result.get("source_documents", []):
                sources.add(os.path.basename(doc.metadata.get("source", "Unknown")))
            if sources:
                print(f"(Source: {', '.join(sources)})\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
