"""AppTek diarization test-set adapter and reproducible WER/CER/DER evaluation."""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


class AppTekDataset:
    DATASET_ID = "apptek-com/apptek_callcenter_dialogues"

    @staticmethod
    def stream(limit: int | None = None):
        """Stream complete mixed-speaker calls; do not download all 873 at once."""
        from datasets import Audio, load_dataset
        from itertools import islice
        ds = load_dataset(AppTekDataset.DATASET_ID, "diarization", split="test", streaming=True)
        # Avoid TorchCodec decoding dependencies; raw WAV bytes stay available.
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(decode=False))
        return islice(ds, limit) if limit is not None else iter(ds)

    @staticmethod
    def audio_to_file(row: dict[str, Any], folder: str | Path) -> Path:
        """Export one HF audio row to an actual local file for WhisperX."""
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        audio = row.get("audio")
        if not isinstance(audio, dict):
            raise ValueError("AppTek row has no embedded audio; use AppTekAudioLoader.load(row) to fetch WAV by file_name.")
        origin = str(audio.get("path") or row.get("file_name") or "call.wav")
        out = path / (Path(origin).stem + ".wav")
        if audio.get("bytes") is not None:
            out.write_bytes(audio["bytes"])
        elif audio.get("path") and Path(audio["path"]).is_file():
            import shutil
            shutil.copyfile(audio["path"], out)
        else:
            raise ValueError("Streaming row has neither WAV bytes nor a local file path.")
        return out


class AppTekAudioLoader:
    """Download the public WAV matching an AppTek diarization row.

    Dataset rows may contain audio=None; file_name is relative to the
    accent-specific directory in the dataset repository. Downloads are cached
    by huggingface_hub, and exact suffix matching prevents wrong-accent files.
    """

    def __init__(self, repo_id: str = AppTekDataset.DATASET_ID,
                 local_dir: str | Path = "data/apptek_audio"):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self._paths: list[str] | None = None

    def load(self, row: dict[str, Any]) -> Path:
        from huggingface_hub import HfApi, hf_hub_download

        relative_name = str(row.get("file_name") or "").replace("\\", "/").lstrip("/")
        if not relative_name or ".." in Path(relative_name).parts:
            raise ValueError(f"Invalid AppTek file_name: {relative_name!r}")
        if self._paths is None:
            self._paths = HfApi().list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        matches = [
            name for name in self._paths
            if name.startswith("diarization/") and name.endswith("/" + relative_name)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one diarization WAV for {relative_name!r}; "
                f"found {len(matches)}: {matches[:5]}"
            )
        return Path(hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=matches[0],
            local_dir=str(self.local_dir),
        ))


class CallEvaluator:
    """Reference comparison using full-call text and raw diarization intervals.

    WER/CER: concatenated reference/predicted segment text in chronological order.
    DER: pyannote optimal speaker permutation, collar=0, overlap included.
    Roles: reference roles only; model speakers mapped for analysis, never inference.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    def _normalize(self, text: str) -> str:
        if not self.normalize:
            return text.strip()
        # Simple transparent normalization (NOT AppTek's full official scorer).
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _annotation(segments: list[dict], speaker_field: str) -> Annotation:
        ann = Annotation()
        for i, s in enumerate(segments):
            start, end = float(s["start"]), float(s["end"])
            if end <= start:
                continue
            label = s.get(speaker_field)
            if label is None or label == "UNKNOWN":
                continue
            ann[Segment(start, end), i] = str(label)
        return ann

    def evaluate(self, prediction: dict[str, Any], reference_row: dict[str, Any]) -> dict[str, Any]:
        ref_segments = sorted(reference_row["segments"], key=lambda s: (s["start"], s["end"]))
        hyp_segments = sorted(prediction["segments"], key=lambda s: (s["start"], s["end"]))
        ref_text = self._normalize(" ".join(str(s.get("text", "")) for s in ref_segments))
        hyp_text = self._normalize(" ".join(str(s.get("text", "")) for s in hyp_segments))
        if not ref_text:
            raise ValueError("Reference transcript is empty; WER cannot be computed.")

        reference = self._annotation(ref_segments, "speaker_id")
        hypothesis = self._annotation(prediction.get("diarization", []), "speaker")
        if not prediction.get("diarization"):
            raise ValueError("No raw diarization spans in prediction; cannot compute DER correctly.")
        details = self.der_metric.compute_components(reference, hypothesis)
        der = self.der_metric.compute_metric(details)
        # Store raw edit counts so the aggregate WER/CER is truly corpus-level.
        word_ops = jiwer.process_words(ref_text, hyp_text)
        char_ops = jiwer.process_characters(ref_text, hyp_text)
        def counts(ops):
            return {
                "hits": int(ops.hits), "substitutions": int(ops.substitutions),
                "deletions": int(ops.deletions), "insertions": int(ops.insertions),
                "reference_units": int(ops.hits + ops.substitutions + ops.deletions),
            }
        # Reference roles are only used for post-hoc analysis; no role labels inferred here.
        return {
            "call_id": prediction.get("call_id"),
            "reference_file": reference_row.get("file_name"),
            "domain": reference_row.get("domain"),
            "accent": reference_row.get("accent"),
            "wer": float(word_ops.wer),
            "cer": float(char_ops.cer),
            "der": float(der),
            "word_counts": counts(word_ops),
            "char_counts": counts(char_ops),
            "reference_words": len(ref_text.split()),
            "hypothesis_words": len(hyp_text.split()),
            "der_components_seconds": {k: round(float(v), 3) for k, v in details.items()},
            "processing_seconds": prediction.get("processing_seconds"),
            "duration_seconds": prediction.get("duration_seconds"),
            "rtf": round(prediction["processing_seconds"] / prediction["duration_seconds"], 4)
                if prediction.get("duration_seconds") and prediction.get("processing_seconds") is not None else None,
            "evaluation_notes": [
                "WER/CER use simple lowercase/punctuation normalization, not the official AppTek normalization pipeline.",
                "WER is whole-call sequential text; overlapping speech may affect comparison.",
                "DER uses raw Pyannote diarization spans, optimal anonymous-speaker matching, no collar, overlap included.",
                "No Customer/Agent identity accuracy is claimed without a separately validated role-mapping step.",
            ],
        }

    @staticmethod
    def save_metrics(metrics: list[dict[str, Any]], output_dir: str | Path) -> None:
        import pandas as pd
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(metrics).drop(columns=["der_components_seconds", "evaluation_notes", "word_counts", "char_counts"], errors="ignore").to_csv(
            out / "metrics.csv", index=False
        )
        # The summary is written by AppTekEvaluationRunner after all calls finish.

    @staticmethod
    def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Pooled/corpus-level rates, never an average of per-call percentages.

        WER/CER aggregate raw jiwer edit counts. DER aggregates pyannote
        components (seconds) before computing its ratio. RTF is duration-weighted.
        """
        if not metrics:
            return {"wer": None, "cer": None, "der": None, "rtf": None}

        def pooled_edits(key: str) -> tuple[float | None, dict[str, int]]:
            keys = ("hits", "substitutions", "deletions", "insertions", "reference_units")
            total = {k: 0 for k in keys}
            for row in metrics:
                counts = row.get(key)
                if not isinstance(counts, dict):
                    raise ValueError(f"Missing {key}. Re-evaluate calls using updated evaluation.py.")
                for k in keys:
                    total[k] += int(counts[k])
            denom = total["reference_units"]
            rate = ((total["substitutions"] + total["deletions"] + total["insertions"]) / denom
                    if denom else None)
            return rate, total

        wer, words = pooled_edits("word_counts")
        cer, chars = pooled_edits("char_counts")

        der_keys = ("total", "confusion", "false alarm", "missed detection")
        seconds = {k: 0.0 for k in der_keys}
        for row in metrics:
            components = row.get("der_components_seconds")
            if not isinstance(components, dict):
                raise ValueError("Missing DER components. Re-evaluate calls using updated evaluation.py.")
            for k in der_keys:
                seconds[k] += float(components.get(k, 0))
        der = ((seconds["confusion"] + seconds["false alarm"] + seconds["missed detection"])
               / seconds["total"] if seconds["total"] > 0 else None)

        valid_times = [m for m in metrics if m.get("processing_seconds") is not None
                       and m.get("duration_seconds") is not None and m["duration_seconds"] > 0]
        total_processing = sum(float(m["processing_seconds"]) for m in valid_times)
        total_duration = sum(float(m["duration_seconds"]) for m in valid_times)
        rtf = total_processing / total_duration if total_duration else None
        return {
            "wer": wer, "cer": cer, "der": der, "rtf": rtf,
            "word_counts": words, "character_counts": chars,
            "der_components_seconds": seconds,
            "processing_seconds": total_processing,
            "audio_duration_seconds": total_duration,
            "calls_with_timing": len(valid_times),
            "notes": [
                "Pooled WER/CER from all reference words/characters, not call averages.",
                "Pooled DER from summed Pyannote components; collar=0 and overlap included.",
                "DER uses Pyannote's inferred UEM, not an explicit full-audio scoring window.",
                "WER/CER use simple lowercase/punctuation normalization, not official AppTek scoring.",
                "Only successful calls are included; failed calls are listed separately.",
            ],
        }


class AppTekEvaluationRunner:
    """Batch evaluation: stream -> download WAV -> transcribe -> score -> save.

    The runner can use any transcriber with a transcribe(path, output_path)
    method returning the standard prediction dictionary.
    """

    def __init__(self, transcriber: Any, evaluator: CallEvaluator | None = None,
                 audio_loader: AppTekAudioLoader | None = None):
        self.transcriber = transcriber
        self.evaluator = evaluator or CallEvaluator(normalize=True)
        self.audio_loader = audio_loader or AppTekAudioLoader()

    def run(self, limit: int = 10, output_dir: str | Path = "results",
            rows: Any = None, reuse_predictions: bool = True) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        metrics: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        source = rows if rows is not None else AppTekDataset.stream(limit=limit)

        for i, row in enumerate(source, start=1):
            if i > limit:
                break
            file_name = str(row.get("file_name") or f"call_{i:03d}.wav")
            try:
                output_path = target / f"{i:03d}_{Path(file_name).stem}.json"
                if reuse_predictions and output_path.is_file():
                    prediction = json.loads(output_path.read_text(encoding="utf-8"))
                    print(f"[{i}/{limit}] Reusing saved prediction: {output_path.name}")
                else:
                    audio_path = self.audio_loader.load(row)
                    prediction = self.transcriber.transcribe(audio_path, output_path)
                score = self.evaluator.evaluate(prediction, row)
                # Keep text references for independent, offline review in Streamlit.
                # Reference audio is NOT included or redistributed here.
                ref_path = target / "reference_transcripts" / f"{Path(file_name).stem}.json"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                ref_path.write_text(json.dumps({
                    "call_id": Path(file_name).stem, "file_name": file_name,
                    "segments": [{"start": seg.get("start"), "end": seg.get("end"),
                                  "speaker": seg.get("speaker_id"), "text": seg.get("text", "")}
                                 for seg in row.get("segments", [])],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                metrics.append(score)
                print(f"[{i}/{limit}] WER={score['wer']:.3f} | "
                      f"CER={score['cer']:.3f} | DER={score['der']:.3f} | "
                      f"RTF={score['rtf'] if score['rtf'] is not None else 'N/A'}")
            except Exception as exc:
                failures.append({
                    "call_index": i, "file_name": file_name,
                    "error_type": type(exc).__name__, "error": str(exc),
                })
                print(f"[{i}/{limit}] FAILED {file_name}: {exc}")

        # Never represent an empty experiment as a successful benchmark.
        self.evaluator.save_metrics(metrics, target)
        (target / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metric_names = ("wer", "cer", "der", "rtf")
        summary = {
            "requested_calls": limit,
            "successful_calls": len(metrics),
            "failed_calls": len(failures),
            "aggregation": "pooled rates are primary; macro mean/std retained for diagnostics",
            "overall": self.evaluator.aggregate(metrics),
            "mean": {}, "std": {},
        }
        for name in metric_names:
            values = [float(m[name]) for m in metrics if m.get(name) is not None]
            summary["mean"][name] = statistics.mean(values) if values else None
            summary["std"][name] = statistics.stdev(values) if len(values) > 1 else None
        (target / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nOVERALL (pooled across successful calls):")
        for name in metric_names:
            value = summary["overall"][name]
            print(f"  {name.upper()}: {value:.4f}" if value is not None else f"  {name.upper()}: N/A")
        print(f"Completed: {len(metrics)} successful / {len(failures)} failed")
        return {"summary": summary, "metrics": metrics, "failures": failures}