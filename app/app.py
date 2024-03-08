import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Setting up OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "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"

# Function to read text from PDF
def read_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    raw_text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to split text into chunks
def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

# Function to perform document search
def perform_document_search(texts):
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search

# Function to perform question answering
def perform_question_answering(docs, query):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

# Streamlit app
def main():
    # Set background image using CSS
    st.markdown(
        f"""
        <style>
            body {{
                background-color:skyblue;
                
            }}
            .stApp {{
                background-color:skyblue;
                margin: auto;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Document Search and Question Answering")
    
    # File uploader for PDF
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.subheader("PDF Preview")
        raw_text = read_pdf(uploaded_file)
        st.header("UPLOADED SUCCESSFULLY 👍")
        st.subheader("Document Search")
        texts = split_text(raw_text)
        document_search = perform_document_search(texts)
        
        # Question input
        query = st.text_input("Ask a Question")
        
        if st.button("Search"):
            if query:
                st.subheader("Search Results")
                docs = document_search.similarity_search(query)
                result = perform_question_answering(docs, query)
                st.write(result)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
