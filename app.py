import streamlit as st
from inquire_docs import (
    extract_text_from_pdf,
    split_text_to_chunks,
    create_vector_store,
    load_qa_model
)
from langchain.chains import RetrievalQA

# Configure page
st.set_page_config(page_title="Inquire Document Q&A", layout="centered")
st.title("📄 Inquire document Q&A App")
st.markdown("Upload a PDF and ask questions. The app will find answers using AI + Vector Search.")

# Upload section
uploaded_file = st.file_uploader("📎 Upload a PDF document", type=["pdf"])
question = st.text_input("❓ Ask a question about the document:")

# Caching document processing
@st.cache_data(show_spinner=False)
def process_document(file_obj):
    file_obj.seek(0)  # ✅ reset file pointer
    raw_text = extract_text_from_pdf(file_obj)
    chunks = split_text_to_chunks(raw_text)
    vectorstore = create_vector_store(chunks)
    return vectorstore

# QA Button Logic
if st.button("🔍 Get Answer"):
    if uploaded_file is None:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤖 Processing document and fetching answer..."):
            try:
                vectorstore = process_document(uploaded_file)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

                llm = load_qa_model()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff"
                )

                answer = qa_chain.run(question)
                st.success("✅ Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
