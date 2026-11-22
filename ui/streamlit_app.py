# Streamlit chat UI. Talks to the FastAPI backend over HTTP; shows the approval panel and results.
import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Kaggle Baseline Agent")
# session_state survives Streamlit reruns (every click reruns this whole script).
state = st.session_state
state.setdefault("messages", [])  # list of (role, text)
state.setdefault("thread_id", None)
state.setdefault("approval", None)  # what the agent wants approved, or None
state.setdefault("run_id", None)


def send(path: str, payload: dict) -> None:
    """POST to the API and store what came back: a reply, or an approval request."""
    with st.spinner("Agent is working..."):
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=600)
    if response.status_code != 200:
        st.error(response.json().get("detail", response.text))
        return
    result = response.json()
    state.thread_id = result["thread_id"]
    state.approval = result["approval"]
    state.run_id = result["run_id"] or state.run_id
    if result["reply"]:
        state.messages.append(("assistant", result["reply"]))


def resume(decision: dict) -> None:
    send(f"/chat/{state.thread_id}/resume", decision)
    st.rerun()


def show_approval() -> None:
    """Show each proposed tool call with editable args, and the three decision buttons."""
    st.warning("The agent wants to run this. Approve, edit, or reject?")
    edited_text = {}
    for call in state.approval["tool_calls"]:
        st.markdown(f"**{call['name']}**")
        edited_text[call["id"]] = st.text_area("args", json.dumps(call["args"], indent=2), key=call["id"])
    feedback = st.text_input("Feedback (used if you reject)")

    approve, edit, reject = st.columns(3)
    if approve.button("Approve"):
        resume({"action": "approve"})
    if edit.button("Edit & approve"):
        try:
            args = {call_id: json.loads(text) for call_id, text in edited_text.items()}
        except json.JSONDecodeError as error:
            st.error(f"Args are not valid JSON: {error}")
        else:
            resume({"action": "edit", "args": args})
    if reject.button("Reject"):
        resume({"action": "reject", "feedback": feedback or "No reason given."})


def show_results() -> None:
    """Show the metrics table of the latest saved run."""
    run = requests.get(f"{API_URL}/runs/{state.run_id}", timeout=30).json()
    st.subheader("Results")
    st.caption(f"Dataset: {run['dataset']} | Target: {run['target']} | Best model: {run['best_model']}")
    st.dataframe(pd.DataFrame(run["results"]))


for role, text in state.messages:
    st.chat_message(role).write(text)

if state.approval:
    show_approval()
elif state.run_id:
    show_results()

if prompt := st.chat_input("e.g. Build a baseline model for house prices", disabled=bool(state.approval)):
    state.messages.append(("user", prompt))
    state.run_id = None
    send("/chat", {"message": prompt, "thread_id": state.thread_id})
    st.rerun()
