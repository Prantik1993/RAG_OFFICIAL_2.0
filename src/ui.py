"""
Streamlit UI for Legal RAG System
"""

import streamlit as st
import requests
import uuid
import os

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"

# Page configuration
st.set_page_config(
    page_title="GDPR Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);}
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1>⚖️ GDPR Legal Assistant v3.0</h1>
    <p style='color: #666;'>Powered by LLM-driven query analysis and smart retrieval</p>
</div>
""", unsafe_allow_html=True)

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("### 💡 Example Queries")
    
    with st.expander("🎯 Exact Lookups", expanded=True):
        examples = [
            "What is Article 15.1.a?",
            "Show me recital 42",
            "Display Article 6"
        ]
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    with st.expander("📊 Range Queries"):
        examples = [
            "What articles are in Chapter 2 Section 3?",
            "Chapter 5 starts from which article?",
            "Show all articles in Chapter 3 Section 1"
        ]
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    with st.expander("💭 Conceptual Questions"):
        examples = [
            "What are the consent requirements?",
            "What rights do data subjects have?",
            "Explain the principles of data processing"
        ]
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    try:
        response = requests.get(STATS_ENDPOINT, timeout=2)
        if response.status_code == 200:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Initializing...")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    
    # Session info
    st.markdown("### 🔐 Session")
    st.caption(f"**ID:** `{st.session_state.session_id[:16]}...`")
    st.caption(f"**Messages:** {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.markdown("### 💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Show metadata for assistant responses
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            
            with st.expander("📚 View Sources", expanded=False):
                if meta.get("sources"):
                    st.caption(f"**Pages:** {', '.join(map(str, meta['sources']))}")
                
                if meta.get("reference_paths"):
                    st.caption("**References:**")
                    for ref in meta["reference_paths"]:
                        st.caption(f"  • {ref}")

# Handle example query
if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
else:
    prompt = st.chat_input("Ask about GDPR regulations...")

# Process query
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing your query..."):
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    metadata = data.get("metadata", {})
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display metadata
                    with st.expander("📚 View Sources", expanded=False):
                        if sources:
                            st.caption(f"**Pages:** {', '.join(map(str, sources))}")
                        
                        if metadata.get("reference_paths"):
                            st.caption("**References:**")
                            for ref in metadata["reference_paths"]:
                                st.caption(f"  • {ref}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "sources": sources,
                            "reference_paths": metadata.get("reference_paths", [])
                        }
                    })
                
                elif response.status_code == 400:
                    error_detail = response.json().get("detail", "Invalid request")
                    st.error(f"⚠️ **Error:** {error_detail}")
                
                else:
                    st.error(f"⚠️ **Error {response.status_code}:** {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Failed:** Backend API is offline. Start it with `uvicorn src.api:app --reload`")
            except requests.exceptions.Timeout:
                st.error("⏱️ **Timeout:** Request took too long")
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔒 Built with LangChain + OpenAI GPT-4o-mini + HuggingFace Embeddings + FAISS</p>
</div>
""", unsafe_allow_html=True)
