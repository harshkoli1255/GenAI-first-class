from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="MirAI Student Policy Advisor", page_icon="🎓", layout="wide")


def get_backend_status() -> dict:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            payload = response.json()
            return {"ok": True, "details": payload}
        return {"ok": False, "details": {"error": f"HTTP {response.status_code}: {response.text}"}}
    except requests.RequestException as exc:
        return {"ok": False, "details": {"error": str(exc)}}


def clear_chat() -> None:
    st.session_state.chat_history = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("MirAI Student Policy Advisor")
st.caption("Ask student policy questions using the official handbook only. The assistant does not invent policies.")

with st.sidebar:
    st.header("System status")
    backend_status = get_backend_status()
    if backend_status["ok"]:
        st.success("Backend online")
        st.json(backend_status["details"])
    else:
        st.error("Backend offline")
        st.write(backend_status["details"].get("error", "Unable to reach the backend."))

    st.markdown("---")
    st.header("Handbook ingestion")
    uploaded_file = st.file_uploader("Upload the official PDF", type=["pdf"], key="pdf_uploader")
    if st.button("Ingest handbook", disabled=not uploaded_file or not backend_status["ok"]):
        with st.spinner("Uploading and indexing the PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
                payload = response.json()
                if response.status_code == 200:
                    st.success(payload.get("message", "Handbook ingested successfully."))
                else:
                    st.error(payload.get("detail", "The handbook could not be ingested."))
            except requests.RequestException as exc:
                st.error(f"Request failed while ingesting the PDF: {exc}")

    if st.button("Clear chat"):
        clear_chat()

for chat_item in st.session_state.chat_history:
    with st.chat_message(chat_item["role"]):
        st.write(chat_item["content"])
        if chat_item.get("sources"):
            with st.expander("Source references"):
                for source in chat_item["sources"]:
                    if source.get("page") is not None:
                        st.write(f"- {source['source']} (page {source['page']})")
                    else:
                        st.write(f"- {source['source']}")

prompt = st.chat_input("Ask about attendance, leave, clubs, or campus policy rules...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    if not backend_status["ok"]:
        response_text = "The backend is offline or unavailable. Please start the FastAPI server and try again."
        st.session_state.chat_history.append({"role": "assistant", "content": response_text, "sources": []})
        with st.chat_message("assistant"):
            st.write(response_text)
    else:
        with st.spinner("Reading the relevant policy sections..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": prompt},
                    timeout=120,
                )
                payload = response.json()
                if response.status_code == 200:
                    answer = payload.get("answer", "No answer was returned by the backend.")
                    sources = payload.get("source_references", [])
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                    with st.chat_message("assistant"):
                        st.write(answer)
                        if sources:
                            with st.expander("Source references"):
                                for source in sources:
                                    if source.get("page") is not None:
                                        st.write(f"- {source['source']} (page {source['page']})")
                                    else:
                                        st.write(f"- {source['source']}")
                else:
                    detail = payload.get("detail", "The backend could not answer that question.")
                    st.session_state.chat_history.append({"role": "assistant", "content": detail, "sources": []})
                    with st.chat_message("assistant"):
                        st.write(detail)
            except requests.RequestException as exc:
                error_text = f"The request to the backend failed: {exc}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_text, "sources": []})
                with st.chat_message("assistant"):
                    st.write(error_text)
