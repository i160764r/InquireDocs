# rag-doc-chat
Ask questions about your documents using GPT + vector search
This project lets you upload a PDF or text file, and then ask natural language questions about its content. It uses RAG (Retrieval-Augmented Generation) to fetch relevant chunks and generate smart, accurate answers.

Top Features:

- Upload PDF or TXT documents
- Ask context-aware questions
- Uses OpenAI (or other LLMs) for answers
- Stores embeddings in FAISS
- Simple Streamlit UI
- Easily extensible (e.g., add citations or multiple docs)

How it Works:
- A[User Uploads PDF] --> B[Text Extractor]
- B --> C[Text Chunking]
- C --> D[Embeddings]
- D --> E[FAISS Vector DB]
- F[User Asks Question] --> G[Convert Q to Vector]
- G --> H[Retrieve Similar Chunks from FAISS]
- H --> I[Send to LLM (GPT)]
- I --> J[Answer Generated]
- J --> K[Display in UI]

Tech Stack:
Layer	          Tool
UI	            Streamlit
LLM             OpenAI (GPT-4)
Embeddings	    OpenAI / SentenceTransformers
Vector DB       FAISS
Text Parsing	  PyMuPDF
Orchestration	  LangChain






