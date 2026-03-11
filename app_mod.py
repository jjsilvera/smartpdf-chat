import os
import streamlit as st
from rag_utility_mod import process_document_to_chroma_db, answer_question, reset_vectorstore
from langchain.memory import ConversationBufferMemory


# Number of files allowed
MAX_FILES = 3
st.set_page_config(page_title="SmartPDF Chatbot", page_icon="📕", layout="wide")
st.title("🚀 SmartPDF Chat")

# Initilialize session
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
# Sidebar
with st.sidebar:
    st.header("⚙️ Search settings")
    k = st.slider("Number of similar documents to retrieve (k)", min_value=1, max_value=10, value=4, step=1,
    help = "Adjust the number of similar documents retrieved for answering questions. A higher k may provide more context but can also introduce noise.")
    temperature = st.slider("Temperature for response generation", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    help = "Adjust the creativity of the generated responses. A higher temperature may produce more diverse answers, while a lower temperature will make the output more focused and deterministic.")
    search_type = st.selectbox(
        "Select search type for retrieving similar documents",
        options=["similarity", "mmr"],
        index=0,
        help = "'similarity' retrieves documents based on cosine similarity, while 'mmr' (Maximal Marginal Relevance) balances relevance and diversity in the retrieved documents."
    )
    st.markdown("---")    
    if st.button("Reset all data and start fresh"):
        reset_vectorstore()
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.show_reset_toast = True
        st.rerun()
        
    if st.session_state.get("show_reset_toast"):
        st.toast("✅ All data has been reset. You can now upload new documents.") # toast
        st.session_state.show_reset_toast = False
        
    if st.session_state.processed_files:
        st.markdown('### 📄 Processed Files')
        for f in st.session_state.processed_files:
            st.write(f"- {f}")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        f"Upload PDF documents (max {MAX_FILES} files)",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"❌ You can upload a maximum of {MAX_FILES} files. Please remove {len(uploaded_files) - MAX_FILES} file(s).")
        else:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            file_names = [f.name for f in uploaded_files]
            st.write(f"**Uploaded Files:** {', '.join(file_names)}")
            
            if st.button("Process Uploaded Files"):
                processed_count = 0
                progress_bar = st.progress(0)
                status_container = st.container()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            if process_document_to_chroma_db(uploaded_file):
                                processed_count += 1
                                st.toast(f"✅ {uploaded_file.name} uploaded successfully!", icon="✅")
                                
                                with status_container:
                                    st.info(f"✅ {uploaded_file.name} processed successfully!")
                                                                    
                    except Exception as e:
                        with status_container:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                if processed_count > 0:
                    for f in file_names:
                        if f not in st.session_state.processed_files:
                            st.session_state.processed_files.append(f)
                    with status_container:
                        st.success(f"{processed_count} file(s) processed and added to the database!")
                else:
                    with status_container:
                        st.warning("⚠️ No files were processed successfully. Please check the error messages and try again.")
                
with col2:
    st.subheader("💬 Ask Questions")
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_files" in message:
                with st.expander("**Source Files**"):
                    st.write(", ".join(message["source_files"]))
    
    # User input
    if prompt := st.chat_input("Type your question here..."):
        if not st.session_state.processed_files:
            st.warning("⚠️ Please upload and process at least one PDF document before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_question(
                    user_question=prompt,
                    k=k,
                    search_type=search_type,
                    temperature=temperature,
                    memory=st.session_state.memory
                    )
                    answer = response["answer"]
                    source_docs = response["source_documents"]
                    source_files = set()
                    for doc in source_docs:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            source_files.add(doc.metadata['source'])
                    st.markdown(answer)
                    
                    if source_files:
                        with st.expander("📚 Source Files"):
                            st.write(f"**Files:** {', '.join(source_files)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_files": list(source_files)
                    })
                    

        
        
                
        



