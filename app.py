"""
CampusPilot — AI College Assistant
Modern dark-themed Streamlit front end for the LangGraph RAG pipeline.
"""

import os
import time
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CampusPilot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — dark, modern, production-grade
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #14192b 0%, #0a0d16 45%, #060810 100%);
        color: #e6e8ef;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Force sidebar to stay permanently open, no collapse toggle ---------- */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    button[data-testid="baseButton-headerNoPadding"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
        transform: none !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 300px !important;
        max-width: 300px !important;
        margin-left: 0px !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10131f 0%, #0b0e18 100%);
        border-right: 1px solid rgba(124, 138, 255, 0.12);
    }
    section[data-testid="stSidebar"] * {
        color: #cfd3e6 !important;
    }

    .brand-mark {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .brand-mark .logo-dot {
        width: 34px; height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #7c8aff, #b06bff);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
        box-shadow: 0 0 24px rgba(124,138,255,0.45);
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f4f5fb;
    }
    .brand-sub {
        font-size: 12.5px;
        color: #8890ab;
        margin-top: -4px;
        margin-bottom: 18px;
        letter-spacing: 0.2px;
    }

    .sidebar-section-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: #6f7796;
        font-weight: 600;
        margin: 18px 0 8px 0;
    }

    /* Programme pills via radio */
    div[role="radiogroup"] label {
        background: rgba(124, 138, 255, 0.06);
        border: 1px solid rgba(124, 138, 255, 0.15);
        border-radius: 10px;
        padding: 8px 12px !important;
        margin-bottom: 6px;
        transition: all 0.15s ease;
    }
    div[role="radiogroup"] label:hover {
        border-color: rgba(124, 138, 255, 0.5);
        background: rgba(124, 138, 255, 0.12);
    }

    .status-card {
        background: rgba(124, 138, 255, 0.06);
        border: 1px solid rgba(124, 138, 255, 0.15);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 13px;
        margin-top: 8px;
    }
    .status-row {
        display: flex; justify-content: space-between;
        padding: 3px 0;
        color: #aab0cc;
    }
    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; background: #4ade80;
        margin-right: 6px; box-shadow: 0 0 8px #4ade80;
    }

    /* ---------- Header ---------- */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 4px 18px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 18px;
    }
    .app-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #a8b1ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        margin: 2px 0 0 0;
        color: #8890ab;
        font-size: 13.5px;
    }
    .programme-badge {
        background: linear-gradient(135deg, rgba(124,138,255,0.18), rgba(176,107,255,0.18));
        border: 1px solid rgba(124,138,255,0.35);
        color: #d6d9ff;
        padding: 7px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    /* ---------- Chat bubbles ---------- */
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 4px 0;
    }
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 16px;
        line-height: 1.55;
        font-size: 14.5px;
        max-width: 100%;
    }
    .bubble-user {
        background: linear-gradient(135deg, #2b2f52, #232744);
        border: 1px solid rgba(124,138,255,0.25);
        color: #eef0ff;
    }
    .bubble-ai {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e2e4f1;
    }
    .tag-chip {
        display: inline-block;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 999px;
        margin-bottom: 8px;
    }
    .tag-academic { background: rgba(124,138,255,0.16); color: #a9b3ff; border: 1px solid rgba(124,138,255,0.35); }
    .tag-fee { background: rgba(255,183,94,0.14); color: #ffcd8a; border: 1px solid rgba(255,183,94,0.35); }
    .tag-general { background: rgba(78,222,166,0.12); color: #7fe3bb; border: 1px solid rgba(78,222,166,0.3); }

    /* ---------- Chat input ---------- */
    div[data-testid="stChatInput"] {
        background: #10131f;
        border: 1px solid rgba(124,138,255,0.25);
        border-radius: 14px;
    }
    div[data-testid="stChatInput"] textarea {
        color: #f0f1fa !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #6f7dff, #a35bff);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 8px 16px;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(124,138,255,0.35);
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0d16; }
    ::-webkit-scrollbar-thumb { background: #2a2f4d; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# STATE DEFINITION (same as your LangGraph pipeline)
# ----------------------------------------------------------------------------
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ----------------------------------------------------------------------------
# CACHED RESOURCES — retrievers + LLM only built once
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_retriever(pdf_path: str):
    embeddings = load_embeddings()
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)


@st.cache_resource(show_spinner=False)
def build_graph():
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
    llm = get_llm()

    def classifier_node(state: State) -> dict:
        last_message = state["messages"][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )
        response = llm.invoke(prompt)
        category = response.content.strip().lower()
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )
        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    def route_query(state: State):
        if state["query_type"] == "academic":
            return "academic_rag"
        elif state["query_type"] == "fee":
            return "fee_rag"
        else:
            return "general"

    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: role, content, query_type
if "programme" not in st.session_state:
    st.session_state.programme = "BCA"

PROGRAMME_OPTIONS = ["BCA", "BBA", "B.Com (H)"]

TAG_CLASS = {
    "academic": ("tag-academic", "ACADEMIC"),
    "fee": ("tag-fee", "FEE & PAYMENTS"),
    "general": ("tag-general", "GENERAL"),
}

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
            <div class="logo-dot">🧭</div>
            <div class="brand-title">CampusPilot</div>
        </div>
        <div class="brand-sub">Your AI-powered campus assistant</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Your Programme</div>', unsafe_allow_html=True)
    selected = st.radio(
        "Programme",
        PROGRAMME_OPTIONS,
        index=PROGRAMME_OPTIONS.index(st.session_state.programme),
        label_visibility="collapsed",
    )
    st.session_state.programme = selected

    st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-row"><span><span class="status-dot"></span>LLM engine</span><span>Groq · Llama 3.3 70B</span></div>
            <div class="status-row"><span><span class="status-dot"></span>Vector store</span><span>FAISS</span></div>
            <div class="status-row"><span><span class="status-dot"></span>Knowledge base</span><span>Handbook + Fees</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1>CampusPilot</h1>
            <p>Ask about attendance, exams, fees, scholarships, or anything campus-related.</p>
        </div>
        <div class="programme-badge">🎓 {st.session_state.programme} Student</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CHAT HISTORY RENDER
# ----------------------------------------------------------------------------
if not st.session_state.chat_history:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color:#6f7796;">
            <div style="font-size: 42px; margin-bottom: 10px;">👋</div>
            <div style="font-size: 16px; font-weight:600; color:#c7cbe4;">Welcome to CampusPilot</div>
            <div style="font-size: 13.5px; margin-top:4px;">Try asking: "What is the attendance requirement?" or "When is the last date to pay semester fees?"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for entry in st.session_state.chat_history:
    role = entry["role"]
    with st.chat_message("user" if role == "user" else "assistant"):
        if role == "assistant" and entry.get("query_type") in TAG_CLASS:
            cls, label = TAG_CLASS[entry["query_type"]]
            st.markdown(f'<span class="tag-chip {cls}">{label}</span>', unsafe_allow_html=True)
        bubble_cls = "bubble-user" if role == "user" else "bubble-ai"
        st.markdown(f'<div class="chat-bubble {bubble_cls}">{entry["content"]}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------------------------------
user_query = st.chat_input("Message CampusPilot...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble bubble-user">{user_query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("CampusPilot is thinking..."):
            try:
                app = build_graph()
                result = app.invoke(
                    {
                        "programme": st.session_state.programme,
                        "messages": [("human", user_query)],
                    }
                )
                answer = result["messages"][-1].content
                q_type = result.get("query_type", "general")
            except Exception as e:
                answer = f"Something went wrong while processing your question: `{e}`"
                q_type = "general"

        if q_type in TAG_CLASS:
            cls, label = TAG_CLASS[q_type]
            st.markdown(f'<span class="tag-chip {cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble bubble-ai">{answer}</div>', unsafe_allow_html=True)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "query_type": q_type}
    )