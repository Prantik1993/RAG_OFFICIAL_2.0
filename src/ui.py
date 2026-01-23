import streamlit as st
import requests
import uuid
import os

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"

# Page Setup
st.set_page_config(
    page_title="GDPR Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚖️ GDPR Legal Assistant v2.0")
    st.markdown("*Advanced RAG system with intelligent retrieval routing*")
with col2:
    try:
        stats = requests.get(STATS_ENDPOINT, timeout=2).json()
        if stats.get("indexed"):
            st.metric("Articles Indexed", stats.get("total_articles", "N/A"))
    except:
        pass

# Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with examples
with st.sidebar:
    st.header("💡 Example Queries")
    
    st.subheader("Exact References")
    if st.button("What is Article 15.1.a?"):
        st.session_state.example_query = "What is Article 15.1.a?"
    if st.button("Show me Article 6"):
        st.session_state.example_query = "Show me Article 6"
    
    st.subheader("Conceptual Questions")
    if st.button("What are consent requirements?"):
        st.session_state.example_query = "What are the consent requirements under GDPR?"
    if st.button("Tell me about data subject rights"):
        st.session_state.example_query = "What rights do data subjects have?"
    
    st.subheader("Comparisons")
    if st.button("Compare Article 6 and 7"):
        st.session_state.example_query = "What's the difference between Article 6 and Article 7?"
    
    st.divider()
    
    st.caption("**Session ID:**")
    st.caption(f"`{st.session_state.session_id[:16]}...`")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            col1, col2 = st.columns(2)
            with col1:
                if metadata.get("sources"):
                    st.caption(f"📄 **Pages:** {', '.join(map(str, metadata['sources']))}")
            with col2:
                if metadata.get("query_type"):
                    st.caption(f"🔍 **Type:** {metadata['query_type']}")
            
            if metadata.get("references"):
                with st.expander("📚 Referenced Sections"):
                    for ref in metadata["references"]:
                        st.markdown(f"- {ref}")

# Handle example query click
if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
else:
    prompt = st.chat_input("Ask about GDPR regulations...")

# Process Query
if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing regulations..."):
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
                    query_type = data.get("query_type", "general")
                    metadata_info = data.get("metadata", {})
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        if sources:
                            st.caption(f"📄 **Pages:** {', '.join(map(str, sources))}")
                    with col2:
                        st.caption(f"🔍 **Type:** {query_type}")
                    
                    # Show referenced articles
                    if metadata_info.get("references"):
                        with st.expander("📚 Referenced Sections"):
                            for ref in metadata_info["references"]:
                                st.markdown(f"- {ref}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "sources": sources,
                            "query_type": query_type,
                            "references": metadata_info.get("references", [])
                        }
                    })
                
                else:
                    error_msg = f"⚠️ Error {response.status_code}: {response.text}"
                    st.error(error_msg)
            
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Failed:** Backend API is not running. Start it with `uvicorn src.api:app --reload`")
            except Exception as e:
                st.error(f"❌ **Error:** {e}")

# Footer
st.divider()
st.caption("🔒 This system uses advanced hybrid retrieval for accurate legal information extraction.")