\# RAG Document Chatbot



A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your own documents (PDF, DOCX, TXT). Built using Google Gemini, LangChain, and ChromaDB.



\## Features



\- 📄 Supports PDF, DOCX, and TXT documents

\- 🤖 Powered by Google Gemini AI (via `gemini-flash-latest`)

\- 🔍 Uses vector embeddings for accurate semantic search

\- 💾 Persistent vector storage using ChromaDB

\- 🌐 Multilingual support (English and Tamil)

\- 📚 Shows source document references for every answer



\## Tech Stack



\- \*\*Language:\*\* Python

\- \*\*LLM:\*\* Google Gemini (`gemini-flash-latest`)

\- \*\*Embeddings:\*\* Google Generative AI Embeddings (`gemini-embedding-001`)

\- \*\*Framework:\*\* LangChain

\- \*\*Vector Database:\*\* ChromaDB

\- \*\*Document Loaders:\*\* PyPDFLoader, Docx2txtLoader, TextLoader



\## How It Works



1\. Documents are loaded from the `documents/` folder

2\. Text is split into chunks using `RecursiveCharacterTextSplitter`

3\. Each chunk is converted into vector embeddings

4\. Embeddings are stored in a local ChromaDB vector store

5\. On each query, the most relevant chunks are retrieved

6\. Gemini generates an answer using the retrieved context



\## Setup



\### Prerequisites

\- Python 3.10+

\- Google Gemini API key (\[get one here](https://aistudio.google.com/app/apikey))



\### Installation



```bash

\# Clone the repository

git clone <your-repo-url>

cd RAG-Document-Chatbot



\# Create and activate a virtual environment

python -m venv venv

venv\\Scripts\\activate   # Windows



\# Install dependencies

pip install -r requirements.txt

```



\### Configuration



Create a `.env` file in the project root:



```

GOOGLE\_API\_KEY=your\_api\_key\_here

```



\### Add Documents



Place your PDF, DOCX, or TXT files inside the `documents/` folder.



\### Run



```bash

python main.py

```



Ask questions in the terminal. Type `exit` or `quit` to stop.



\## Project Structure



```

RAG-Document-Chatbot/

├── documents/          # Your source documents (PDF/DOCX/TXT)

├── chroma\_db/           # Auto-generated vector database

├── main.py               # Main application

├── requirements.txt   # Python dependencies

├── .env                     # API key (not committed to git)

└── README.md

```



\## Example



```

Neenga: What are Kishore's skills?



Bot: Based on the provided document, Kishore is currently learning

Python and building a RAG chatbot using Gemini and LangChain.



(Source: test.txt)

```



\## Future Improvements



\- \[ ] Web-based UI using Streamlit

\- \[ ] Conversation memory for follow-up questions

\- \[ ] Support for more file types (CSV, HTML)

\- \[ ] Deployment to cloud



\## License



This project is for educational purposes.

