import streamlit as st
import pdfplumber
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS FOR FIXED UI ----------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}
h1, h2, h3, h4 {
    color: #a78bfa !important;
}
[data-testid="stSidebar"] .stSelectbox, 
[data-testid="stSidebar"] .stButton,
[data-testid="stFileUploader"],
[data-testid="stTextInput"] {
    color: #e5e7eb !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
}
.message-bubble.user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 14px 16px;
    border-radius: 12px;
    margin: 10px 0;
    max-width: 75%;
    margin-left: auto;
}
.message-bubble.bot {
    background: #1f2937;
    color: #e5e7eb;
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid #374151;
    margin: 10px 0;
    max-width: 75%;
    margin-right: auto;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 1rem;
    left: 320px;
    right: 1rem;
    background: #1f2937;
    border-radius: 12px;
    border: 1px solid #374151;
    padding: 8px 16px;
    color: white;
}
[data-testid="stChatInput"] textarea {
    color: #f3f4f6 !important;
    background: none !important;
}
[data-testid="stVerticalBlock"] {
    padding-right: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BACKEND ----------------
OLLAMA_BASE = "http://localhost:11434"  # ✅ Fixed: plain URL, not Markdown

def ask_ollama(prompt, model="llama3"):
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 300}
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json().get("response", "⚠️ No response from Ollama.")
        else:
            return f"⚠️ Ollama error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama is not running. Please start it with: `ollama serve`"
    except Exception as e:
        return f"⚠️ Error connecting to Ollama: {str(e)}"

def get_models():
    EXCLUDED = ["embed", "nomic", "minilm", "bge", "e5", "gte"]
    try:
        res = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        models = [m["name"] for m in res.json().get("models", [])]
        models = [m for m in models if not any(k in m.lower() for k in EXCLUDED)]
        return models if models else ["llama3"]
    except:
        return ["llama3"]

# ---------------- STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None
if "pdf" not in st.session_state:
    st.session_state.pdf = None

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## DocMind 🧠")
file = st.sidebar.file_uploader("📄 Upload your PDF", type="pdf")
model = st.sidebar.selectbox("🤖 Select Model", get_models())

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat = []

# ---------------- MAIN TITLE ----------------
st.markdown("<h1 style='text-align:center;'>DocMind AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9ca3af;'>Ask your documents anything with the power of AI ⚡</p>", unsafe_allow_html=True)

# ---------------- PDF PROCESS ----------------
if file and st.session_state.pdf != file.name:
    with st.spinner("🔍 Extracting content from PDF..."):
        text = ""
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_text(text)
        emb = load_embeddings()
        st.session_state.db = FAISS.from_texts(chunks, emb)
        st.session_state.pdf = file.name
        st.success("✅ PDF processed and ready for questions!")

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.chat:
    st.markdown(f'<div class="message-bubble user">{msg["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message-bubble bot">{msg["a"]}</div>', unsafe_allow_html=True)

# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask something about your document...")

if query:
    if not st.session_state.db:
        st.warning("📁 Please upload a PDF first.")
    else:
        with st.spinner("Thinking... 🤔"):
            docs = st.session_state.db.similarity_search(query, k=2)
            context = "\n\n".join([d.page_content[:500] for d in docs])
            prompt = f"""Answer using the context below in a concise and clear way.

Context:
{context[:1500]}

Question: {query}

Answer:"""
            answer = ask_ollama(prompt, model)
            st.session_state.chat.append({"q": query, "a": answer})
            st.rerun()