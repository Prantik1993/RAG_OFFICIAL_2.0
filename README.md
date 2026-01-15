Markdown

# ⚖️ GDPR Legal Assistant (RAG)

A production-grade **Retrieval-Augmented Generation (RAG)** application designed to answer legal questions based on the **General Data Protection Regulation (GDPR)**. 

This project uses a modular architecture with **FastAPI** (Backend) and **Streamlit** (Frontend), featuring **Hybrid Search** (Vector Search + Reranking) and **Inline Citations** for high accuracy.

---

## 🚀 Features

* **📚 Document Ingestion:** Automatically loads, chunks, and indexes PDF legal documents.
* **🧠 Hybrid Search:** Uses **FAISS** for fast retrieval and **FlashRank** for precise reranking.
* **🤖 Context-Aware AI:** Uses OpenAI (GPT-3.5/4) to generate answers with strict adherence to the provided context.
* **📝 Inline Citations:** Answers include precise page references (e.g., *"[Page 42]"*) for legal verification.
* **🛡️ Guardrails:** Validates user inputs to prevent hallucinations or abuse.
* **🔌 API-First Design:** Fully decoupled backend (FastAPI) and frontend (Streamlit).
* **🐳 Docker Ready:** Includes a Dockerfile for easy containerized deployment.

---

## 🛠️ Tech Stack

* **LLM:** OpenAI GPT-3.5 Turbo
* **Orchestration:** LangChain
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) - *Runs locally & free*
* **Reranker:** FlashRank (`ms-marco-MiniLM-L-12-v2`) - *Runs locally & free*
* **Backend:** FastAPI
* **Frontend:** Streamlit

---

## 📂 Project Structure

1. Prerequisites
Python 3.10+

OpenAI API Key

2. Clone & Install
Bash

# Clone the repository
git clone https://github.com/Prantik1993/RAG_OFFICIAL_2.0.git

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Environment Configuration
Create a .env file in the root directory:
OPENAI_API_KEY=sk-proj-your-actual-key-here...
4. Add Data
Ensure your PDF file is placed in the data/ folder

(Note: The system will automatically build the database on the first run)

🏃‍♂️ How to Run
You need to run the Backend and Frontend in separate terminals.

Terminal 1: Backend (API)
This starts the RAG Engine and API Server.

uvicorn src.api:app --reload
Wait until you see: Application startup complete

Terminal 2: Frontend (UI)
This launches the web interface.

streamlit run src/ui.py
Your browser should automatically open to http://localhost:8501

🐳 Running with Docker
If you prefer not to install Python locally, use Docker.

Build the Image:

Bash

docker build -t gdpr-rag .
Run the Container:

Bash

docker run -p 8000:8000 -e OPENAI_API_KEY=your-key-here gdpr-rag
🔍 How It Works (The Pipeline)
Ingestion: The app checks if storage/faiss_index exists. If not, it loads the PDF from data/, splits it into chunks (preserving Article numbers), embeds them using HuggingFace, and saves the index.

Retrieval: When you ask a question, the system fetches the top 20 most similar chunks.

Reranking: FlashRank re-orders those 20 chunks to find the top 3 that actually answer the question.

Generation: The LLM receives the top 3 chunks and generates an answer with inline citations (e.g., [Page 45]).

🛡️ License
This project is for educational purposes.