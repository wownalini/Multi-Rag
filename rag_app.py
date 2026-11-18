import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG App")
st.caption("Upload multiple PDFs and ask questions with sources")

# File uploader
files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if files and "store" not in st.session_state:
    with st.spinner("Indexing documents..."):
        all_chunks = []
        
        for uploaded_file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Add source filename to metadata
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            
            # Clean up temp file
            os.unlink(tmp_path)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        st.session_state.store = FAISS.from_documents(all_chunks, embeddings)
        
        st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} file(s)")

# Question answering
if "store" in st.session_state:
    question = st.text_input("Ask a question about your documents:")
    
    if question:
        with st.spinner("Searching..."):
            # Retrieve relevant chunks
            retriever = st.session_state.store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)
            
            # Generate answer
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""Answer the question based on the context below. 
If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
            
            response = llm.invoke(prompt)
            
            st.subheader("Answer")
            st.write(response.content)
            
            st.subheader("Sources")
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                st.caption(f"📄 {source} — page {page}")
else:
    st.info("Upload one or more PDFs to get started.")