import mimetypes

import streamlit as st
from dotenv import load_dotenv

from backend.api_client import APIClientError, PapeerAPIClient

load_dotenv()

st.set_page_config(page_title="Papeer", page_icon="📚", layout="centered")


@st.cache_resource
def get_api_client() -> PapeerAPIClient:
    return PapeerAPIClient()


def create_session(api: PapeerAPIClient) -> str:
    session = api.create_session()
    sid = session["id"]
    st.session_state.sessions_meta[sid] = session
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(api: PapeerAPIClient, session_id: str) -> list[dict]:
    try:
        return api.get_history(session_id)
    except Exception as exc:
        st.error(f"Could not restore chat history for this session: {exc}")
        return []


def switch_session(api: PapeerAPIClient, session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(api, session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


api = get_api_client()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
try:
    sessions = api.list_sessions()
    st.session_state.sessions_meta = {session["id"]: session for session in sessions}
except APIClientError as exc:
    st.error(f"Papeer API unavailable: {exc}")
    st.session_state.sessions_meta = st.session_state.get("sessions_meta", {})
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if (
    "active_session_id" not in st.session_state
    or st.session_state.active_session_id not in st.session_state.sessions_meta
):
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(api, latest["id"])
    else:
        try:
            st.session_state.active_session_id = create_session(api)
        except APIClientError as exc:
            st.error(f"Could not create a session: {exc}")
            st.stop()

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        try:
            new_sid = create_session(api)
        except APIClientError as exc:
            st.error(f"Could not create a session: {exc}")
            st.stop()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(api, sid)
                st.rerun()

    st.divider()
    st.markdown("## 📄 Documents")  

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error("Upload no more than five files at a time.")
                st.stop()
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                files_to_upload = [
                    f for f in uploaded_files
                    if f.name not in st.session_state[processed_key]
                ]
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                if not files_to_upload:
                    st.info("All selected files have already been loaded.")
                else:
                    try:
                        results = api.upload_documents(
                            active_sid,
                            [
                                (
                                    f.name,
                                    f.getvalue(),
                                    f.type or mimetypes.guess_type(f.name)[0] or "application/octet-stream",
                                )
                                for f in files_to_upload
                            ],
                        )
                        for result in results:
                            if result["ok"]:
                                st.session_state[processed_key].add(result["filename"])
                                st.success(f"Added: {result['filename']}")
                            else:
                                st.error(f"Failed: {result['filename']} — {result['error']}")
                    except APIClientError as exc:
                        st.error(f"File upload failed: {exc}")
            st.rerun()

        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            if len(urls) > 5:
                st.error("Load no more than five URLs at a time.")
                st.stop()
            with st.spinner("Loading web pages…"):
                try:
                    results = api.load_urls(active_sid, urls)
                    for result in results:
                        if result["ok"]:
                            st.success(f"Loaded: {result['url'][:60]}")
                        else:
                            st.error(f"Failed: {result['url'][:60]} — {result['error']}")
                except APIClientError as exc:
                    st.error(f"URL loading failed: {exc}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    result = api.load_arxiv(active_sid, arxiv_title.strip())
                    st.success(f"Loaded: {result['title']}")
                except APIClientError as exc:
                    st.error(f"ArXiv loading failed: {exc}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
    try:
        doc_titles = api.list_documents(active_sid)
    except APIClientError as exc:
        doc_titles = None
        st.caption(f"Could not load document list: {exc}")
    if doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    elif doc_titles == []:
        st.caption("No documents loaded yet.")


    
   

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📚 Papeer — Research Paper Assistant")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload papers in the sidebar and start chatting below."
)
st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────

for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                try:
                    for event in api.btw_stream(query):
                        if event["event"] == "token":
                            response_text += event["data"].get("text", "")
                            placeholder.markdown(response_text + "▌")
                        elif event["event"] == "error":
                            raise APIClientError(502, event["data"].get("detail", "API error"))
                    placeholder.markdown(response_text or "No response generated.")
                except APIClientError as exc:
                    st.error(f"/btw request failed: {exc}")
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""
            state_snapshot = {}
            completed = False
            try:
                for event in api.chat_stream(active_sid, prompt):
                    if event["event"] == "token":
                        response_text += event["data"].get("text", "")
                        placeholder.markdown(response_text + "▌")
                    elif event["event"] == "done":
                        response_text = event["data"].get("answer") or response_text
                        state_snapshot = event["data"].get("graph_state", {})
                        completed = True
                    elif event["event"] == "error":
                        raise APIClientError(
                            502,
                            event["data"].get("detail", "Chat request failed."),
                        )
                if not completed:
                    raise APIClientError(502, "The API stream ended before completion.")
                placeholder.markdown(response_text or "No response generated.")
            except APIClientError as exc:
                st.error(f"Chat request failed: {exc}")

            if completed:
                with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                    st.json(state_snapshot)
                st.session_state.chats[active_sid].append(
                    {
                        "role": "assistant",
                        "content": response_text or "No response generated.",
                        "graph_state": state_snapshot,
                        "turn": current_turn,
                    }
                )