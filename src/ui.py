import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
# In a real deployment, this URL might come from an environment variable
API_BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="GDPR Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ GDPR Legal Assistant")
st.markdown("ask questions about the *General Data Protection Regulation*.")

# --- SESSION MANAGEMENT ---
# Generate a unique ID for this browser tab so the API tracks conversation history correctly
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY HISTORY ---
# Re-render previous messages every time the page refreshes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT HANDLER ---
if prompt := st.chat_input("Ask about Article 6, Consent, or Fines..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Communicate with Backend API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Regulation..."):
            try:
                # Prepare Payload
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
                
                # Send Request
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources (if any)
                    if sources:
                        st.caption(f"**Reference Pages:** {', '.join(map(str, sources))}")
                    
                    # Save context to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    # Handle API Errors (e.g., 400 Bad Request, 500 Server Error)
                    error_msg = f"⚠️ Error {response.status_code}: {response.text}"
                    st.error(error_msg)
            
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Failed:** Could not connect to the Backend API. Is `src/api.py` running?")
            except Exception as e:
                st.error(f"❌ **An unexpected error occurred:** {e}")

# --- SIDEBAR (Optional Debug Info) ---
with st.sidebar:
    st.header("System Status")
    try:
        # Simple health check (pinging the docs endpoint)
        status = requests.get(f"{API_BASE_URL}/docs", timeout=1)
        if status.status_code == 200:
            st.success("Backend Online ✅")
        else:
            st.warning("Backend Unstable ⚠️")
    except:
        st.error("Backend Offline 🔴")
        
    st.divider()
    st.caption(f"Session ID:\n`{st.session_state.session_id}`")