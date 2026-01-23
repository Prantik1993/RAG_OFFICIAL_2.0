import streamlit as st
import requests
import uuid
import os
from datetime import datetime

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"

# Page Setup with modern styling
st.set_page_config(
    page_title="GDPR Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern chat UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat container */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: white;
        border-left: 4px solid #667eea;
    }
    
    /* Input box */
    .stChatInputContainer {
        background-color: white;
        border-radius: 24px;
        border: 2px solid #e9ecef;
        padding: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    /* Code blocks */
    code {
        background-color: #f7fafc;
        color: #667eea;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f7fafc;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with modern design
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='margin: 0; font-size: 2.5rem;'>⚖️ GDPR Legal Assistant</h1>
            <p style='color: #718096; margin-top: 0.5rem; font-size: 1.1rem;'>
                AI-powered legal document analysis with hybrid retrieval
            </p>
        </div>
    """, unsafe_allow_html=True)

# Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = None

# Sidebar with examples and controls
with st.sidebar:
    st.markdown("### 💡 Example Queries")
    
    # Quick examples in expandable sections
    with st.expander("🎯 Exact References", expanded=True):
        if st.button("📄 What is Article 15.1.a?", use_container_width=True):
            st.session_state.example_query = "What is Article 15.1.a?"
        if st.button("📄 Show me Article 6", use_container_width=True):
            st.session_state.example_query = "Show me Article 6"
        if st.button("📄 Recital 42", use_container_width=True):
            st.session_state.example_query = "What is Recital 42?"
        if st.button("📄 Regulation point no 15", use_container_width=True):
            st.session_state.example_query = "Regulation point no 15"
    
    with st.expander("🤔 Conceptual Questions"):
        if st.button("💭 Consent requirements", use_container_width=True):
            st.session_state.example_query = "What are the consent requirements under GDPR?"
        if st.button("💭 Data subject rights", use_container_width=True):
            st.session_state.example_query = "What rights do data subjects have?"
        if st.button("💭 Lawful basis for processing", use_container_width=True):
            st.session_state.example_query = "What are the lawful bases for processing personal data?"
    
    with st.expander("⚖️ Comparisons"):
        if st.button("🔄 Article 6 vs Article 7", use_container_width=True):
            st.session_state.example_query = "What's the difference between Article 6 and Article 7?"
        if st.button("🔄 Consent vs Legitimate Interest", use_container_width=True):
            st.session_state.example_query = "Compare consent and legitimate interest as legal bases"
    
    st.markdown("---")
    
    # System stats
    st.markdown("### 📊 System Status")
    try:
        stats = requests.get(STATS_ENDPOINT, timeout=2).json()
        if stats.get("indexed"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📚 Articles", stats.get("total_articles", "N/A"))
            with col2:
                st.metric("📝 Recitals", stats.get("total_recitals", "N/A"))
            
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ Indexing in progress...")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    
    # Session info
    st.markdown("### 🔐 Session Info")
    st.caption(f"**ID:** `{st.session_state.session_id[:16]}...`")
    
    if st.session_state.last_query_time:
        st.caption(f"**Last Query:** {st.session_state.last_query_time}")
    
    st.caption(f"**Messages:** {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query_time = None
        st.rerun()

# Main chat area
st.markdown("### 💬 Chat")

# Display Chat History with modern styling
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Show metadata for assistant responses
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            # Display in a nice compact format
            cols = st.columns([1, 1, 2])
            
            with cols[0]:
                if metadata.get("sources"):
                    pages_str = ", ".join(map(str, metadata["sources"][:5]))
                    if len(metadata["sources"]) > 5:
                        pages_str += f" +{len(metadata['sources']) - 5} more"
                    st.caption(f"📄 **Pages:** {pages_str}")
            
            with cols[1]:
                if metadata.get("query_type"):
                    type_emoji = {
                        "recital": "📋",
                        "article": "📄",
                        "point": "🔹",
                        "subsection": "📑",
                        "general": "💭"
                    }
                    emoji = type_emoji.get(metadata["query_type"], "🔍")
                    st.caption(f"{emoji} **Type:** {metadata['query_type'].title()}")
            
            with cols[2]:
                if metadata.get("references"):
                    with st.expander("📚 View References", expanded=False):
                        for ref in metadata["references"][:5]:
                            st.markdown(f"- {ref}")

# Handle example query click
if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
else:
    prompt = st.chat_input("Ask about GDPR regulations...", key="chat_input")

# Process Query
if prompt:
    # Update last query time
    st.session_state.last_query_time = datetime.now().strftime("%H:%M:%S")
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing regulations..."):
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
                    
                    # Display answer with typing effect
                    st.markdown(answer)
                    
                    # Display metadata in compact format
                    cols = st.columns([1, 1, 2])
                    
                    with cols[0]:
                        if sources:
                            pages_str = ", ".join(map(str, sources[:5]))
                            if len(sources) > 5:
                                pages_str += f" +{len(sources) - 5} more"
                            st.caption(f"📄 **Pages:** {pages_str}")
                    
                    with cols[1]:
                        type_emoji = {
                            "recital": "📋",
                            "article": "📄",
                            "point": "🔹",
                            "subsection": "📑",
                            "general": "💭"
                        }
                        emoji = type_emoji.get(query_type, "🔍")
                        st.caption(f"{emoji} **Type:** {query_type.title()}")
                    
                    with cols[2]:
                        if metadata_info.get("references"):
                            with st.expander("📚 View References", expanded=False):
                                for ref in metadata_info["references"][:5]:
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
                
                elif response.status_code == 400:
                    error_detail = response.json().get("detail", "Invalid request")
                    st.error(f"⚠️ **Input Error:** {error_detail}")
                
                else:
                    st.error(f"⚠️ **Error {response.status_code}:** {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Failed:** Backend API is not running. Start it with `uvicorn src.api:app --reload`")
            except requests.exceptions.Timeout:
                st.error("⏱️ **Timeout:** Request took too long. Please try again.")
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #718096; padding: 1rem;'>
        <p style='margin: 0;'>🔒 Powered by Advanced Hybrid Retrieval | Built with Streamlit & LangChain</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
            Using GPT-3.5 Turbo + Sentence Transformers + FlashRank
        </p>
    </div>
""", unsafe_allow_html=True)