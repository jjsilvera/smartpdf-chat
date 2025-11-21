import os
import streamlit as st
from rag_utility_mod import process_document_to_chroma_db, answer_question


# Number of files allowed
MAX_FILES = 3
st.set_page_config(page_title="SmartPDF Chatbot", page_icon="📕", layout="centered")
st.title("🚀 SmartPDF Chat ")

# Initilialize session
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# File uploader widget
uploaded_files = st.file_uploader(
        f"Upload PDF documents (max {MAX_FILES} files)",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader" #
        )

# Process uploaded files
if uploaded_files is not None and len(uploaded_files) > 0:
    if len(uploaded_files) > MAX_FILES:
        st.error(f"❌ You can upload a maximum of {MAX_FILES} files. Please remove {len(uploaded_files) - MAX_FILES} file(s).")
        st.stop()
    else:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # File names
        file_names = [file.name for file in uploaded_files]
        st.write(f"**Uploaded files:** {', '.join(file_names)}")
        
        # Process each file
        processed_count = 0
        all_documents = []
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Process file directly from memory
                    result = process_document_to_chroma_db(uploaded_file)
                    if result:
                        processed_count += 1
                        st.info(f"{uploaded_file.name} processed successfully!")
                        
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Session state
        if processed_count > 0:
            st.session_state.documents_processed = True
            st.session_state.processed_files = file_names[:processed_count]
            st.success(f"🎉 {processed_count} document(s) processed successfully! You can now ask questions.")
        else:
            st.error("❌ No documents were processed successfully.")
            
# Question answering section
if st.session_state.documents_processed:
    st.markdown("---")
    user_question = st.text_area("Ask your question about the documents:")
    
    if st.button("Answer"):
        if user_question.strip():
            with st.spinner("🔍 Searching documents and generating answer..."):
                response = answer_question(user_question)
                answer = response["answer"]
                source_documents = response["source_documents"]
            
            st.markdown("### Response")
            st.markdown(answer)
            # Source documents info
            if source_documents:
                source_files = set()
                for doc in source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source_file = doc.metadata['source']
                        source_files.add(source_file)
                
                if source_files:
                    st.markdown("### Source Files")
                    st.success(f"**Generated from:** {', '.join(sorted(source_files))}")
                                   
        else:
            st.warning("Please enter a question first.")
else:
    st.info("👆 Upload PDF files to start asking questions")


