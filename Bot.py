# Install once:
# pip install langchain langchain-community sentence-transformers faiss-cpu pypdf transformers

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Step 1: Load PDF
loader = PyPDFLoader("AInotes.pdf")   # replace with your PDF file
docs = loader.load()

# Step 2: Create embeddings + FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Step 3: Local text-generation model
qa_model = pipeline("text-generation", model="google/flan-t5-small")

print("Chatbot ready! Type your questions (type 'exit' to quit).\n")

# Step 4: Chat loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Chatbot closed.")
        break

    # Retrieve relevant chunks
    results = db.similarity_search(query, k=1)
    context = " ".join([d.page_content for d in results])

    # Generate detailed answer

    prompt = f"Question: {query}\nContext: {context}\nGive a short answer:"
    answer = qa_model(prompt, max_new_tokens=256, do_sample=False)


    print("Bot:", answer[0]["generated_text"])
