import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

import tempfile
import os

st.title("RAG Document Assistant (LM Studio)")
st.write("Upload a TXT file and ask questions about it.")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="llama-2-7b-chat:2",
    temperature=0.7
)


if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("File uploaded successfully!")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

#tt spliting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    # ===============================
    # EMBEDDINGS
    # ===============================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # ===============================
    # VECTOR DATABASE
    # ===============================

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    # ===============================
    # RAG CHAIN
    # ===============================

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # ===============================
    # USER QUESTION
    # ===============================

    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Thinking..."):

            response = qa_chain.invoke({"query": query})

            st.subheader("Answer")
            st.write(response["result"])