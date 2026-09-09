# Offline PDF RAG Chatbot

An offline chatbot that answers questions from PDF documents using **Langchain**, **FAISS** and **HuggingFace** Transformers.

# Features
- Works fully offline (no API keys required).
- Uses FAISS for vector search.
- HuggingFace `flan-t5-small` for text generation.
- Simple command-line chat loop.

# Installation
```bash
git clone https://github.com/<your-username>/offline-pdf-rag-chatbot.git
cd offline-pdf-rag-chatbot
pip install -r requirements.txt

---

## 📦 Usage

⚡ **Run the chatbot locally:**

```bash
python pdf_chatbot.py

-Replace "AInotes.pdf" in the code with your own PDF file.
-Type your questions in the terminal.
-Type exit to quit the chatbot.
