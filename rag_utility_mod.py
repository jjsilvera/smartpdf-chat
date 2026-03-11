import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
import tempfile
import logging
from dotenv import load_dotenv
import streamlit as st

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate



# Load environment variables
load_dotenv()

# Models
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    """
    Get or create the vector store in session state
    """
    from chromadb.config import Settings
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = Chroma(embedding_function=embedding,
                                    client=None,
                                    client_settings=Settings(anonymized_telemetry=False)
        )
        logger.info("Initialized new vector store in session state")
    return st.session_state.vectordb

def reset_vectorstore():
    """
    Reset the vector store in session state
    """
    if 'vectordb' in st.session_state:
        try:
            st.session_state.vectordb.delete_collection()
        except Exception as e:
            logger.warning(f"Error deleting vector store collection: {str(e)}")
        del st.session_state.vectordb
        logger.info("Vector store reset in session state")
        
    if 'memory' in st.session_state:
        st.session_state.memory.clear()
        logger.info("Conversation memory reset in session state")

def process_document_to_chroma_db(uploaded_file):
    """
    Process a PDF directly from memory without saving local files
    """
    tmp_path = None    
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
        
        vectorstore = get_vectorstore()
        vectorstore.add_documents(filtered_texts)
        return True
        
                
    except Exception as e:
        logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        raise e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
       
def create_retriever(k=4, search_type="similarity"):
    """
    Create a retriever from the vector store in session state
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
def create_conversational_chain(llm, retriever, memory):
    """
    Create a conversational retrieval chain
    """
    prompt_template = """You are a helpful assistant that
    answers questions based on the provided context and chat history.
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and helpful.
    Context: {context}

    Chat History: {chat_history}

    Question: {question}

    Helpful Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
        )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False,
        return_source_documents=True)
    return chain

# Question answering function
def answer_question(user_question, k=4, search_type="similarity", temperature=0.0, memory=None):
    """
    Answer questions using the vector base in memory
    """
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature
        )
        retriever = create_retriever(k=k, search_type=search_type)
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        chain = create_conversational_chain(llm, retriever, memory)
        response = chain.invoke({"question": user_question})
        # Get answer
        return {
            "answer": response["answer"],
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