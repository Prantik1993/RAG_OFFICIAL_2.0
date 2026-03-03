
from __future__ import annotations
import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="GDPR Legal Assistant", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stChatMessage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ GDPR Assistant")
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    st.divider()
    st.subheader("Example queries")

    EXAMPLES = {
        "Exact lookups":    ["What is Article 15?", "Show Article 6.1.f", "Recital 42"],
        "Range queries":    ["Articles in Chapter 3 Section 1?", "What is Chapter 5 about?"],
        "Conceptual":       ["What are consent requirements?", "Rights of data subjects?"],
    }
    for group, items in EXAMPLES.items():
        with st.expander(group):
            for ex in items:
                if st.button(ex, key=ex, use_container_width=True):
                    st.session_state["_pending"] = ex

    st.divider()
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("engine"):
            st.success("API online")
        else:
            st.warning("API initialising…")
    except Exception:
        st.error("API offline")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── header ────────────────────────────────────────────────────────────────────
st.title("⚖️ GDPR Legal Assistant")
st.caption("Ask anything about EU Regulation 2016/679")

# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("refs"):
            with st.expander("Sources"):
                for ref in msg["refs"]:
                    st.caption(f"• {ref}")

# ── input ─────────────────────────────────────────────────────────────────────
prompt = st.session_state.pop("_pending", None) or st.chat_input("Ask about GDPR…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching GDPR…"):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data   = resp.json()
                    answer = data["answer"]
                    refs   = data["metadata"].get("references", [])
                    st.markdown(answer)
                    if refs:
                        with st.expander("Sources"):
                            for ref in refs:
                                st.caption(f"• {ref}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "refs": refs}
                    )
                elif resp.status_code == 429:
                    st.warning(resp.json().get("detail", "Rate limited"))
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:200]}")
            except requests.ConnectionError:
                st.error("Cannot reach API. Run: `uvicorn src.api:app --reload`")
            except Exception as exc:
                st.error(str(exc))
