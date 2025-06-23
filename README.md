# 🏠 Dutch Affordable Rent Act (Wet Betaalbare Huur) RAG Assistant

> **🔍 Demo:** Ask real legal questions about Dutch rent laws with LLM + vector search.  
> Example: _"What rent rules apply to Rijksmonuments in 2024?"_  
> Powered by GPT-4o, LangChain, HuggingFace, and pgvector.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-informational)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green)

---

This is a Retrieval-Augmented Generation (RAG) application designed to answer questions about the Dutch Affordable Rent Act, especially useful for expats who are renting in The Netherlands. It loads Staatsblad PDF documents into a vector database and uses OpenAI's GPT models to answer user questions based on that data.

---

## 🚀 Features

- Loads legal documents from the `documents/` directory
- Embeds text using multilingual HuggingFace embeddings
- Stores embeddings in a PostgreSQL + `pgvector` vector store
- Uses a custom prompt and OpenAI's GPT-4o to answer questions
- CLI and REST API interfaces for flexible querying
- Returns **answer + source snippet** for transparency

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/lilttc/rag-pgvector.git
cd rag_pgvector
```

### 2. Start the PostgreSQL Vector Store with Docker

Ensure Docker is installed, then run:

```bash
docker-compose up --build
```

This spins up a PostgreSQL container with `pgvector` enabled and mounts your SQL initialization script.

### 3. Set Up a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add Your OpenAI API Key

Create a `.env` file in the root directory with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🧠 Usage

### 1. Load Documents into the Vector Store

Ensure your PDF files are in the `documents/` folder, then run:

```bash
python index.py
```

This will:
- Load and split the documents
- Embed the text using a multilingual embedding model
- Store them in the vector database

### 2. Ask a Question via CLI

Query the loaded documents with:

```bash
python cli.py --query "How does the WOZ value influence the rental score? Is there a cap?"
>>> The WOZ value influences the rental score by contributing to the total points a property receives, which can affect its rental price category. There is a cap on the WOZ points to prevent properties from being classified into a higher rental segment solely due to a high WOZ value. This cap is applied to properties with 187 points or more, limiting the WOZ points to a maximum of 33% of the total points. This regulation ensures that properties do not enter the liberalized segment just because of a high WOZ value.

**Source:** Staatsblad 2024, nr. 194, page 65

```
### 3. Interact via REST API

Start the API server:

```bash
uvicorn app:app --reload
```

Then open your browser to:

📎 [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

---

## 🧪 Example Queries

Try these example questions to test the RAG assistant:

1. **How does the energy label (e.g., A++, C, G) affect the point score of a home?**
2. **What is the point value of having a dishwasher or built-in induction stove in a rental unit?**
3. **What percentage increase applies to the maximum rent of a rijksmonument after July 1, 2024?**
4. **How does the WOZ value influence the rental score? Is there a cap?**


---

## 📁 Project Structure

```
.
├── documents/              # PDF source files (Dutch rental laws)
├── pgvector/               # pgvector Docker setup (init.sql, Dockerfile)
├── rag_pipeline/           # Embedder, chain, db setup modules
├── index.py                # Load documents into vectorstore
├── cli.py                  # Query interface (CLI)
├── app.py                  # FastAPI app exposing a REST API
├── .env                    # Store your OPENAI_API_KEY here
├── requirements.txt        # Python dependencies
└── docker-compose.yaml     # PostgreSQL + pgvector container
```

---

## 📌 Notes

- Requires Python 3.11 or lower due to current dependency compatibility
- Tested on Ubuntu with Python 3.11.9 and Docker 24+