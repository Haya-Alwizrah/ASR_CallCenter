"""Local VS Code entrypoint for WhisperX transcription and AppTek evaluation.

Examples:
  python -m src.main transcribe --audio data/sample.wav
  python -m src.main evaluate --limit 10 --reuse
  python -m src.main analyze --input results/sample.json
  python -m src.main analyze --input-dir results --reuse
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def setting(name: str, default: str) -> str:
    return os.getenv(name, default)

def build_transcriber(*, keep_asr_loaded: bool | None = None):
    # Delay heavy imports so ``python -m src.main --help`` works before ML setup.
    from src.pipeline import CallCenterWhisperX

    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token == "hf_your_read_token_here":
        raise RuntimeError("Set HF_TOKEN in your local .env and accept Pyannote model access.")
    device = setting("DEVICE", "auto")
    compute = setting("COMPUTE_TYPE", "auto")
    speakers = setting("NUM_SPEAKERS", "2")

    return CallCenterWhisperX(
        hf_token=token,
        model_name=setting("MODEL_NAME", "small"),
        language=setting("LANGUAGE", "en") or None,
        device=None if device == "auto" else device,
        compute_type=None if compute == "auto" else compute,
        batch_size=int(setting("BATCH_SIZE", "4")),
        num_speakers=None if speakers.lower() in ("auto", "none") else int(speakers),
        keep_asr_loaded=(setting("KEEP_ASR_LOADED", "false").lower() in ("1", "true", "yes")
                            if keep_asr_loaded is None else keep_asr_loaded),
    )

class CallProcessingService:
    """One-call orchestration shared by Streamlit and optional CLI commands."""

    def __init__(self, *, preload: bool = True):
        self.transcriber = build_transcriber(keep_asr_loaded=True)
        if preload:
            self.transcriber.load_models()

    def process(self, audio_path: Path, *, output_dir: Path | None = None) -> dict:
        from src.llm import CallLLM
        out_dir = output_dir or ROOT / setting("RESULTS_DIR", "results")
        out_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = out_dir / f"{audio_path.stem}.json"
        transcript = self.transcriber.transcribe(audio_path, prediction_path)
        analysis_path = out_dir / "summaries" / f"{audio_path.stem}_summary.json"
        # Do not overwrite an existing transcript/summary on API failure.
        analysis = CallLLM().analyze(transcript, analysis_path)
        return {"transcript": transcript, "analysis": analysis,
                "transcript_path": prediction_path, "analysis_path": analysis_path}

    def close(self) -> None:
        self.transcriber.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperX speaker-attributed call analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    p_transcribe = sub.add_parser("transcribe", help="Transcribe one call to JSON")
    p_transcribe.add_argument("--audio", type=Path, required=True)
    p_transcribe.add_argument("--output", type=Path, default=None)
    p_eval = sub.add_parser("evaluate", help="Run pooled AppTek WER/CER/DER/RTF evaluation")
    p_eval.add_argument("--limit", type=int, default=10)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--reuse", action="store_true", help="Reuse matching prediction JSON files")
    p_analysis = sub.add_parser("analyze", help="Summarize + analyze customer sentiment on saved transcripts")
    analysis_input = p_analysis.add_mutually_exclusive_group(required=True)
    analysis_input.add_argument("--input", type=Path, help="Single WhisperX prediction JSON")
    analysis_input.add_argument("--input-dir", type=Path, help="Directory containing transcript JSON files")
    p_analysis.add_argument("--output", type=Path, default=None, help="Single-call output JSON")
    p_analysis.add_argument("--output-dir", type=Path, default=None, help="Batch output folder")
    p_analysis.add_argument("--reuse", action="store_true", help="Reuse analysis JSON if sentiment is already present")
    args = parser.parse_args()
    default_results = ROOT / setting("RESULTS_DIR", "results")

    if args.command == "transcribe":
        output = args.output or default_results / f"{args.audio.stem}.json"
        transcriber = build_transcriber()

        try:
            print("Loading ASR model before transcription...")
            transcriber.load_models()
            result = transcriber.transcribe(args.audio, output)
            print(f"Saved {len(result['segments'])} segments to {output}")
        finally:
            transcriber.close()

    elif args.command == "analyze":
        from src.llm import CallLLM
        service = CallLLM()
        if args.input is not None:
            output = args.output or default_results / "summaries" / f"{args.input.stem}_summary.json"
            result = service.analyze(args.input, output, reuse=args.reuse)
            print(f"Saved summary and sentiment for {result['call_id']} to {output}")
        else:
            output_dir = args.output_dir or default_results / "summaries"
            report = service.analyze_directory(args.input_dir, output_dir, reuse=args.reuse)
            print(json.dumps(report, indent=2))
    else:
        from src.evaluation import AppTekAudioLoader, AppTekEvaluationRunner, CallEvaluator
        transcriber = build_transcriber()

        try:
            # Skip GPU model initialization if all predictions can be reused;
            # transcribe() will lazy-load if a prediction is missing.
            if not args.reuse:
                print("Loading ASR model before evaluation...")
                transcriber.load_models()

            runner = AppTekEvaluationRunner(
                transcriber=transcriber,
                evaluator=CallEvaluator(normalize=True),
                audio_loader=AppTekAudioLoader(
                    local_dir=ROOT / setting("APPTEK_CACHE_DIR", "data/apptek_audio")
                ),
            )

            report = runner.run(
                limit=args.limit,
                output_dir=args.output_dir or default_results,
                reuse_predictions=args.reuse,
            )

            print("\nPooled overall metrics:")
            print(json.dumps(report["summary"]["overall"], indent=2))

            if report["summary"]["successful_calls"] == 0:
                raise RuntimeError("No calls succeeded. Check failures.json.")
        finally:
            transcriber.close()

if __name__ == "__main__":
    main()