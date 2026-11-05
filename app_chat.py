# file: app_chat.py
#
# Streamlit chat UI for the restaurant agent. Talks to the FastAPI server:
#   uvicorn main:app            (terminal 1)
#   streamlit run app_chat.py   (terminal 2)

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Restaurant Assistant", page_icon="🍽️")


@st.cache_data(ttl=600)
def sample_users():
    resp = requests.get(f"{API_URL}/users/sample", params={"n": 25}, timeout=30)
    resp.raise_for_status()
    return resp.json()["users"]


def start_session(user_id):
    resp = requests.post(f"{API_URL}/session", json={"user_id": user_id}, timeout=30)
    resp.raise_for_status()
    st.session_state.session = resp.json()
    st.session_state.messages = []


def show_debug(turn):
    calls = turn.get("tool_calls") or []
    grounded = "✅ grounded" if turn.get("grounded", True) else \
        f"⚠️ not found in tool results: {', '.join(turn.get('ungrounded_names', []))}"
    with st.expander(f"debug · {len(calls)} tool call(s) · {grounded}"):
        if not calls:
            st.caption("No tools were called.")
            return
        table = pd.DataFrame([{"round": c["round"], "tool": c["name"], "args": str(c["args"]),
                               "ms": c["duration_ms"], "chars": c["result_chars"],
                               "error": c["error"] or ""} for c in calls])
        st.dataframe(table, hide_index=True, use_container_width=True)
        for c in calls:
            st.caption(f"{c['name']} → result preview")
            st.code(c["preview"], language="json")

# ==========================
# Sidebar: choose who is chatting
# ==========================

with st.sidebar:
    st.header("Who's chatting?")
    mode = st.radio("User", ["Sample user", "Enter a user ID", "New user"])

    user_id = None
    if mode == "Sample user":
        try:
            users = sample_users()
            labels = {f"{u['user_id']} · {u['home_city']} · {u['review_count']} reviews": u["user_id"]
                      for u in users}
            user_id = labels[st.selectbox("Pick a user", list(labels))]
        except requests.RequestException as e:
            st.error(f"Can't reach the API at {API_URL}: {e}")
    elif mode == "Enter a user ID":
        user_id = st.text_input("Yelp user_id").strip() or None

    if st.button("Start chat", type="primary", use_container_width=True):
        try:
            start_session(user_id)
        except requests.RequestException as e:
            st.error(f"Couldn't start a session: {e}")

    session = st.session_state.get("session")
    if session:
        st.divider()
        if session["user_known"]:
            st.success(f"Chatting as **{session['name'] or session['session_id'][:8]}** "
                       f"({session['home_city']})")
        else:
            st.info("Chatting as a **new user** — the assistant will ask about your tastes.")

# ==========================
# Chat window
# ==========================

st.title("🍽️ Restaurant Assistant")

session = st.session_state.get("session")
if not session:
    st.write("Pick a user in the sidebar and press **Start chat**.")
    st.stop()

for turn in st.session_state.messages:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            show_debug(turn)

if prompt := st.chat_input("Ask for recommendations, reviews, ratings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(f"{API_URL}/chat", timeout=180,
                                     json={"session_id": session["session_id"], "message": prompt})
                if resp.status_code == 404:
                    st.session_state.pop("session")
                    st.error("Session expired (server restarted). Start a new chat in the sidebar.")
                    st.stop()
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                result = {"reply": f"Couldn't reach the API: {e}", "tool_calls": [], "grounded": True}
        turn = {"role": "assistant", "content": result["reply"], **result}
        st.markdown(turn["content"])
        show_debug(turn)
    st.session_state.messages.append(turn)
