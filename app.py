import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.main import CallProcessingService

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
RESULTS = ROOT / os.getenv("RESULTS_DIR", "results")

st.set_page_config(page_title="Call Center Analysis", page_icon="🎧", layout="wide")
st.title("Call Center Agent Performance Analysis")
st.write("Upload a call to see its transcript, summary, and customer sentiment.")


def format_time(seconds):
    if seconds is None:
        return "—"
    minutes, seconds = divmod(float(seconds), 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"

def show_field(title, value):
    st.markdown(f"**{title}**")
    st.write(value or "Not mentioned")

@st.cache_resource(show_spinner=False)
def load_service():
    return CallProcessingService(preload=True)

with st.spinner("Loading speech models..."):
    try:
        service = load_service()
    except Exception as error:
        st.error(f"Could not load models: {error}")
        st.stop()

uploaded = st.file_uploader("Upload a call recording", type=["wav", "mp3", "m4a", "flac", "ogg"])
if uploaded is None:
    st.stop()

content = uploaded.getvalue()
call_id = hashlib.sha256(content).hexdigest()[:16]

if st.session_state.get("call_id") != call_id:
    st.session_state.call_id = call_id
    st.session_state.pop("result", None)
    st.session_state.show_conversation = False

if "result" not in st.session_state:
    try:
        upload_dir = RESULTS / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_path = upload_dir / f"{call_id}{Path(uploaded.name).suffix.lower()}"
        audio_path.write_bytes(content)

        with st.spinner("Transcribing and analyzing the call..."):
            st.session_state.result = service.process(audio_path)
    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.stop()

result = st.session_state.result
transcript = result["transcript"]
analysis = result["analysis"]
summary = analysis["summary"]
sentiment = analysis["sentiment"]

st.caption(f"Duration: {format_time(transcript.get('duration_seconds'))}")
st.subheader("Call Summary")
st.write(summary.get("short_summary") or "No summary available")
left, right = st.columns(2)
with left:
    show_field("Customer Issue", summary.get("customer_issue"))
    show_field("Final Outcome / Next Step", summary.get("final_outcome_or_next_step"))
with right:
    show_field("Agent Action", summary.get("agent_action"))

st.subheader("Customer Sentiment")
st.metric("Overall Sentiment", sentiment.get("label", "Unknown"))
show_field("Reason", sentiment.get("justification"))
left, right = st.columns(2)
left.write(f"**Beginning:** {sentiment.get('beginning', 'Unknown')}")
right.write(f"**End:** {sentiment.get('end', 'Unknown')}")
st.caption("Sentiment is based on the transcript, not the voice tone.")

if st.button("Hide Conversation" if st.session_state.show_conversation else "Show Conversation"):
    st.session_state.show_conversation = not st.session_state.show_conversation

if st.session_state.show_conversation:
    roles = analysis.get("speaker_roles", {})
    names = {speaker: role.title() for role in ("customer", "agent")
             if (speaker := roles.get(role))}
    rows = []
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker") or "UNKNOWN"
        rows.append({
            "Start": format_time(segment.get("start")),
            "End": format_time(segment.get("end")),
            "Speaker": names.get(speaker, speaker),
            "Text": segment.get("text", ""),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Download Transcript (JSON)",
                       json.dumps(transcript, ensure_ascii=False, indent=2),
                       file_name=f"{call_id}_transcript.json", mime="application/json")