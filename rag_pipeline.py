import fitz  # PyMuPDF
from typing import List, Union
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA



def extract_text_from_pdf(file: Union[str, "BytesIO"]) -> str:
    text = ""
    # fitz.open accepts file path or file-like object (BytesIO)
    with fitz.open(stream=file.read() if hasattr(file, "read") else file, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    if hasattr(file, "seek"):
        file.seek(0)  # reset file pointer after read
    return text


def split_text_to_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def create_vector_store(chunks: List[str]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)


def load_qa_model() -> HuggingFacePipeline:
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=512,
        device=-1  # Use 0 for GPU if available
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def get_answer_from_pdf(file, question: str) -> str:
    text = extract_text_from_pdf(file)
    chunks = split_text_to_chunks(text)
    vectorstore = create_vector_store(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = load_qa_model()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa.run(question)
    return answer


if __name__ == "__main__":
    # === STEP 1: PDF Extraction ===
    pdf_path = "docs/resume.pdf"  # Change this path as needed
    print("📄 Loading PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print("✅ Text snippet:\n", raw_text[:300], "...")

    # === STEP 2: Split Text ===
    print("🧩 Splitting text into chunks...")
    chunks = split_text_to_chunks(raw_text)
    print(f"✅ Split into {len(chunks)} chunks.")

    # === STEP 3: Embedding and Vector Store ===
    print("📦 Creating vector store...")
    vectorstore = create_vector_store(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("✅ Vector store created.")

    # === STEP 4: Load Model ===
    print("🤖 Loading Hugging Face QA model...")
    llm = load_qa_model()
    print("✅ Model loaded.")

    # === STEP 5: Ask a Question ===
    question = "What programming languages does the candidate know?"
    print("❓ Question:", question)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa.run(question)
    print("💬 Answer:", answer)
