"""Upload-only Streamlit demo: process one new recording, then display its analysis."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
RESULTS = ROOT / os.getenv("RESULTS_DIR", "results")

st.set_page_config(page_title="Call Center Analysis", page_icon="🎧", layout="wide")
st.title("Call Center Agent Performance Analysis")
st.caption("Upload a new call to generate its transcript, summary and customer sentiment.")


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def time_string(seconds: float | int | None) -> str:
    try:
        minutes, sec = divmod(max(float(seconds), 0), 60)
        return f"{int(minutes):02d}:{sec:05.2f}"
    except (TypeError, ValueError):
        return "—"


def display_field(label: str, value: str | None) -> None:
    st.markdown(f"**{label}**")
    st.write(value if value else "Not established in the conversation")


@st.cache_resource(show_spinner=False)
def get_runtime():
    """Load speech models immediately when the app starts; reuse on reruns."""
    from src.main import CallProcessingService
    return CallProcessingService(preload=True)


with st.spinner("Loading WhisperX, alignment and diarization models..."):
    try:
        runtime = get_runtime()
        startup_error = None
    except Exception as exc:
        runtime = None
        startup_error = str(exc)

if startup_error:
    st.error(f"Model initialization failed: {startup_error}")
    st.info("Check HF_TOKEN, Pyannote permissions, FFmpeg and local model dependencies before uploading audio.")
else:
    st.caption("Speech models ready")

uploaded = st.file_uploader(
    "Upload a call recording", type=["wav", "mp3", "m4a", "flac", "ogg"],
    help="Once uploaded, transcription and analysis start automatically.",
)

if uploaded is None:
    # Never show demo recordings or stored history when somebody first opens the app.
    st.session_state.pop("current_upload", None)
    st.session_state.pop("current_analysis", None)
    st.session_state.pop("current_transcript", None)
    st.session_state.pop("processing_error", None)
    st.session_state.pop("show_conversation", None)
    st.stop()

if runtime is None:
    st.error("Upload cannot be processed until the speech models are ready.")
    st.stop()

payload = uploaded.getvalue()
digest = hashlib.sha256(payload).hexdigest()[:16]
safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(uploaded.name).stem)[:48] or "call"
key = f"{safe_stem}_{digest}"

if st.session_state.get("current_upload") != key:
    st.session_state["current_upload"] = key
    st.session_state.pop("current_analysis", None)
    st.session_state.pop("current_transcript", None)
    st.session_state.pop("processing_error", None)
    st.session_state["show_conversation"] = False

if st.session_state.get("current_transcript") is None or st.session_state.get("current_analysis") is None:
    try:
        if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
            raise RuntimeError("Set GROQ_API_KEY in .env to enable summary and sentiment analysis.")
        upload_dir = RESULTS / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_path = upload_dir / f"{key}{Path(uploaded.name).suffix.lower()}"
        audio_path.write_bytes(payload)
        transcript_path = RESULTS / f"{key}.json"
        analysis_path = RESULTS / "summaries" / f"{key}_summary.json"
        with st.status("Analyzing the uploaded call...", expanded=True) as status:
            transcript = read_json(transcript_path)
            if transcript is None or not isinstance(transcript.get("segments"), list):
                st.write("Transcribing and separating speakers...")
                transcript = runtime.transcriber.transcribe(audio_path, transcript_path)
            analysis = read_json(analysis_path)
            if analysis is None or not all(field in analysis for field in ("summary", "sentiment", "speaker_roles")):
                st.write("Generating summary and customer sentiment...")
                from src.llm import CallLLM
                analysis = CallLLM().analyze(transcript, analysis_path)
            st.session_state["current_transcript"] = transcript
            st.session_state["current_analysis"] = analysis
            st.session_state.pop("processing_error", None)
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as exc:
        st.session_state["processing_error"] = str(exc)

if st.session_state.get("processing_error"):
    st.error(st.session_state["processing_error"])
    if st.button("Retry analysis"):
        st.session_state.pop("processing_error", None)
        st.session_state.pop("current_transcript", None)
        st.session_state.pop("current_analysis", None)
        st.rerun()
    st.stop()

transcript = st.session_state["current_transcript"]
analysis = st.session_state["current_analysis"]
call_id = transcript.get("call_id") or key
st.caption(f"Call ID: {call_id} · Duration: {time_string(transcript.get('duration_seconds'))}")
st.subheader("Call Summary")
summary = analysis.get("summary") or {}
st.write(summary.get("short_summary") or "No summary available")
left, right = st.columns(2)
with left:
    display_field("Customer Issue", summary.get("customer_issue"))
    display_field("Final Outcome / Next Step", summary.get("final_outcome_or_next_step"))
with right:
    display_field("Agent Action", summary.get("agent_action"))

st.subheader("Customer Sentiment")
sentiment = analysis.get("sentiment") or {}
st.metric("Overall Sentiment", sentiment.get("label", "Unknown"))
display_field("Reason", sentiment.get("justification"))
start_col, end_col = st.columns(2)
start_col.write(f"**Beginning:** {sentiment.get('beginning', 'Unknown')}")
end_col.write(f"**End:** {sentiment.get('end', 'Unknown')}")
st.caption("Sentiment is inferred from the transcript text, not acoustic tone.")

# Evidence IDs and technical limitations stay in saved JSON, not the reviewer-facing UI.
if st.button("Hide Conversation" if st.session_state.get("show_conversation") else "Show Conversation",
             type="primary", use_container_width=True):
    st.session_state["show_conversation"] = not st.session_state.get("show_conversation", False)
if st.session_state.get("show_conversation", False):
    roles = analysis.get("speaker_roles") or {}
    label_map = {speaker_id: role.title() for role, speaker_id in roles.items()
                 if role in ("customer", "agent") and speaker_id}
    rows = []
    for i, segment in enumerate(transcript.get("segments", []), 1):
        speaker = segment.get("speaker") or "UNKNOWN"
        rows.append({"#": i, "Start": time_string(segment.get("start")),
                     "End": time_string(segment.get("end")),
                     "Speaker": f"{label_map[speaker]} ({speaker})" if speaker in label_map else speaker,
                     "Text": segment.get("text", "")})
    st.subheader("Segmented Conversation")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Download Transcript (JSON)",
                       data=json.dumps(transcript, indent=2, ensure_ascii=False),
                       file_name=f"{key}_transcript.json", mime="application/json")
