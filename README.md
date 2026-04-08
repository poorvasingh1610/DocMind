# 🧠 DocMind AI

DocMind AI is a PDF Question Answering system built using **Generative AI (RAG - Retrieval Augmented Generation)**.  
It allows users to upload a document and ask questions, and the system answers based on the document content.

---

## 🚀 Features

- 📄 Upload any PDF document  
- 💬 Ask questions in natural language  
- 🧠 Context-based answers using AI  
- ⚡ Fast retrieval using vector database  
- 🎯 More accurate than normal chatbots  

---

## 🧠 How It Works (RAG Pipeline)

1. Extract text from PDF  
2. Split text into chunks  
3. Convert chunks into embeddings  
4. Store embeddings in FAISS vector database  
5. Retrieve relevant chunks using similarity search  
6. Send context + query to LLM (LLaMA via Ollama)  
7. Generate final answer  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **PDF Processing:** pdfplumber  
- **Embeddings:** HuggingFace (MiniLM)  
- **Vector Database:** FAISS  
- **LLM:** Ollama (LLaMA 3)  

---

## ▶️ How to Run Locally

### 1. Start Ollama
```bash
ollama serve
