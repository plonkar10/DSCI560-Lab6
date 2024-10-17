import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

# Modify this line to the name of your PDF file
pdf_file_name = "Ads_cookbook.pdf"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

if __name__ == "__main__":
    # Ensure the file exists
    if not os.path.exists(pdf_file_name):
        print(f"PDF file '{pdf_file_name}' not found in the directory.")
    else:
        # Load the PDF file into a list (in case you want to load multiple PDFs)
        pdf_docs = [pdf_file_name]
        
        # Step 1: Extract text from the PDF
        pdf_text = get_pdf_text(pdf_docs)
        print("\nPDF Text Extracted:")
        print(pdf_text[:1000])  # Printing first 1000 characters to avoid flooding output
        
        # Step 2: Split the text into chunks
        text_chunks = get_text_chunks(pdf_text)
        print("\nNumber of Text Chunks:", len(text_chunks))
        print("First Text Chunk:", text_chunks[0])
        
        # Step 3: Generate and save vector embeddings
        vectorstore = get_vectorstore(text_chunks)
        print("\nVectorstore created successfully!")