import os
import tempfile
import logging
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import streamlit as st
# Load environment variables
load_dotenv()

# Models
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process_document_to_chroma_db(uploaded_file):
    """
    Process a PDF directly from memory without saving local files
    """
    try:
        # Temp file to hold uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} chunks from {uploaded_file.name}")
        
        # Chunks filtering and metadata update
        filtered_texts = []
        for text in texts:
            if len(text.page_content.strip()) > 50:
                # Clean content
                clean_content = ' '.join(text.page_content.split())
                text.page_content = clean_content
                # Add source metadata
                text.metadata['source'] = uploaded_file.name
                filtered_texts.append(text)
        logger.info(f"Filtered to {len(filtered_texts)} valid chunks")
        
        # Get or create vectordb in session state
        if 'vectordb' not in st.session_state:
            # Create new vectordb
            st.session_state.vectordb = Chroma.from_documents(
                documents=filtered_texts,
                embedding=embedding
            )
        else:
            # Add to existing vectordb
            st.session_state.vectordb.add_documents(filtered_texts)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise e

# Question answering function
def answer_question(user_question):
    """
    Answer questions using the vector base in memory
    """
    try:
        # Check vectordb in session state
        if 'vectordb' not in st.session_state:
            return {
                "answer": "No documents processed yet. Please upload and process PDF files first.",
                "source_documents": []
            }
        vectordb = st.session_state.vectordb
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  #
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        response = qa_chain.invoke({"query": user_question})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error answering question: {error_msg}")
             
        if "500" in error_msg or "cloudflare" in error_msg.lower():
            return {
                "answer": "⚠️ The AI service is temporarily unavailable. Please try again in a few moments.",
                "source_documents": []
            }
        else:
            return {
                "answer": f"❌ Error: {error_msg}",
                "source_documents": []
            }