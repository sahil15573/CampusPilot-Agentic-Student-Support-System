# 🧭 CampusPilot

**CampusPilot** is an AI-powered college assistant that answers student queries on academics, fees, and general topics using a **LangGraph-based conditional RAG pipeline**. It classifies each incoming query, routes it to the correct knowledge source (academic handbook or fee structure document), retrieves relevant context using FAISS similarity search, and generates a personalized response based on the student's programme (BCA / BBA / B.Com (H)). The project ships with a modern, dark-themed **Streamlit** frontend built to feel like a production-grade app rather than a typical prototype UI.

## 🌐 Live Demo

🔗 **[CampusPilot live](https://campuspilot-agentic-student-support-system-bhrka626bfsixudszm9.streamlit.app/)**

> Note: The app may take a few seconds to wake up if it has been inactive, as Streamlit Community Cloud puts idle apps to sleep after 12 hours.
---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Running the App](#-running-the-app)
- [How It Works](#-how-it-works)
- [Example Queries](#-example-queries)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

Most college FAQ bots either answer everything from a single knowledge base or rely on brittle keyword matching. CampusPilot instead uses an LLM-based **classifier node** to decide what type of question is being asked, then routes the query through a **LangGraph** conditional graph to the appropriate retrieval pipeline:

- **Academic queries** → retrieved from the academic handbook (attendance, exams, grading, credits, promotion, course structure, degree requirements)
- **Fee queries** → retrieved from the fee structure document (tuition, payment, refunds, late charges, scholarships)
- **General queries** → answered directly by the LLM without retrieval (greetings, casual conversation)

The final response is personalized based on the student's selected programme, so if a document contains figures for multiple programmes, the assistant highlights the one relevant to the student.

---

## ✨ Features

- Conditional RAG routing using LangGraph state graphs
- Separate FAISS vector stores for academic and fee documents
- Programme-aware personalization (BCA / BBA / B.Com (H))
- Query classification with visible category tags in the UI (Academic / Fee / General)
- Modern dark-mode Streamlit interface with custom CSS styling
- Cached embeddings, retrievers, and LLM instances for fast repeated queries
- Session-based chat history with a clear-conversation option
- Groq-hosted Llama 3.3 70B for low-latency inference

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph, conditional edges) |
| LLM Framework | LangChain |
| LLM Provider | Groq — Llama 3.3 70B Versatile |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Document Loading | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Frontend | Streamlit (custom CSS, dark theme) |
| Config | python-dotenv |
| Language | Python 3.11 |

---

## 🏗 Architecture

```
                         ┌─────────────┐
                         │    START    │
                         └──────┬──────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Classifier Node │
                       │ (LLM categorizes │
                       │  the query)      │
                       └────────┬────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
     │ Academic RAG │   │   Fee RAG    │   │   General    │
     │   Node       │   │    Node      │   │    Node      │
     │ (FAISS on    │   │ (FAISS on    │   │ (no retrieval│
     │  handbook)   │   │  fee doc)    │   │  needed)     │
     └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                                ▼
                       ┌─────────────────┐
                       │  Response Node   │
                       │ (LLM generates   │
                       │ personalized     │
                       │ final answer)    │
                       └────────┬────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │     END     │
                         └─────────────┘
```

**State schema:**

```python
class State(TypedDict):
    programme: str                          # BCA / BBA / B.Com (H)
    messages: Annotated[list, add_messages]  # conversation history
    query_type: str                          # academic / fee / general
    retrieved_context: str                   # retrieved chunks or "NO_RETRIEVAL_NEEDED"
```

---

## 📁 Project Structure

```
CampusPilot/
│
├── streamlit_app.py          # Main Streamlit UI + LangGraph pipeline
├── academics_handbook.pdf    # Source document for academic queries
├── fee_structure.pdf         # Source document for fee queries
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/CampusPilot.git
cd CampusPilot
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can generate a free API key at [console.groq.com](https://console.groq.com).

> ⚠️ Never commit your `.env` file. It is already excluded via `.gitignore`.

---

## ▶️ Running the App

Make sure `academics_handbook.pdf` and `fee_structure.pdf` are present in the project root, then run:

```bash
streamlit run streamlit_app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🧠 How It Works

1. **User selects their programme** (BCA, BBA, or B.Com (H)) from the sidebar.
2. **User types a query** in the chat input.
3. The **classifier node** sends the query to the LLM, which returns one of three categories: `academic`, `fee`, or `general`.
4. Based on the category, the **conditional edge** routes the state to:
   - `academic_rag` → retrieves top 4 relevant chunks from the academic handbook FAISS index
   - `fee_rag` → retrieves top 4 relevant chunks from the fee structure FAISS index
   - `general` → skips retrieval entirely
5. The **response node** builds a final prompt (with or without retrieved context) and generates a personalized answer using the student's programme.
6. The answer, along with its classified category, is displayed in the chat UI with a colored tag chip for transparency into the routing decision.

---

## 💬 Example Queries

| Query | Routed To |
|---|---|
| "What is the minimum attendance required to sit for exams?" | Academic RAG |
| "When is the last date to pay semester fees?" | Fee RAG |
| "Is there a scholarship for meritorious students?" | Fee RAG |
| "How many credits are needed for promotion to next year?" | Academic RAG |
| "Hi, how are you?" | General |

---

## 🚀 Future Improvements

- Add persistent chat history storage (SQLite / Supabase)
- Support multi-turn conversational memory across sessions
- Add authentication so students see only their programme's data automatically
- Deploy on Streamlit Community Cloud or Render with cached FAISS indices
- Add source citation snippets (page numbers) alongside answers
- Support additional document types (timetables, exam schedules)

---

## 📄 License

This project is open-sourced for educational and portfolio purposes. Feel free to fork and adapt it.

---

**Built by Sahil** — as part of an end-to-end GenAI/Data project portfolio.
