import streamlit as st
import time

# 1. Page Configuration
st.set_page_config(page_title="Swedish Archives RAG", page_icon="📜")
st.title("📜 Swedish Historical Archives RAG Engine")
st.caption("Powered by Fine-tuned BGE-M3 (LoRA) & Qdrant Vector Database")

# 2. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Welcome! I am the historical archives assistant. You can ask me questions about Swedish court records, police reports, and historical events from the 1860s."
    })

# 3. Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("Enter your search query (e.g., theft cases in 1868)..."):
    
    # Display user input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simulate Backend Processing (Retrieval & Reranking)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Querying Qdrant Database via BGE-M3...*")
        
        # Simulate processing time
        time.sleep(1.5) 
        
        # Mock Response (To be replaced with actual Qdrant + LLM pipeline)
        mock_response = f"""
        **Retrieval Results:**
        
        Based on your query **"{prompt}"**, I retrieved the Top-5 relevant documents from the Qdrant database and applied cross-encoder reranking.
        
        The most relevant hit is from **Volume 30002021 (1868)**, a Police Detective Department report. The record indicates...
        
        *(Note: This is a UI mockup. The backend data engine is not yet connected.)*
        """
        
        message_placeholder.markdown(mock_response)
        
    # Append to chat history
    st.session_state.messages.append({"role": "assistant", "content": mock_response})