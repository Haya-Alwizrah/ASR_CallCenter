from __future__ import annotations
import json
import re
import statistics
from urllib.parse import quote
from pathlib import Path
from typing import Any
import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

class AppTekDataset:
    DATASET_ID = "apptek-com/apptek_callcenter_dialogues"

    @staticmethod
    def stream(limit: int | None = None):
        from datasets import Audio, load_dataset
        from itertools import islice
        ds = load_dataset(AppTekDataset.DATASET_ID, "diarization", split="test", streaming=True)

        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(decode=False))
        return islice(ds, limit) if limit is not None else iter(ds)

    @staticmethod
    def audio_to_file(row: dict[str, Any], folder: str | Path) -> Path:
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
    def __init__(self, repo_id: str = AppTekDataset.DATASET_ID,
                 local_dir: str | Path = "data/apptek_audio"):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self._paths: list[str] | None = None

    def file_path(self, row: dict[str, Any]) -> str:
        from huggingface_hub import HfApi

        name = str(row.get("file_name") or "").replace("\\", "/").lstrip("/")
        if not name or ".." in Path(name).parts:
            raise ValueError(f"Invalid audio file name: {name}")
        if self._paths is None:
            self._paths = HfApi().list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        matches = [p for p in self._paths if p.startswith("diarization/") and p.endswith("/" + name)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Cannot find unique audio file for {name}")
        return matches[0]

    def audio_url(self, row: dict[str, Any]) -> str:
        filename = self.file_path(row)
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{quote(filename, safe='/')}"

    def load(self, row: dict[str, Any]) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=self.file_path(row),
            local_dir=str(self.local_dir),
        ))


class CallEvaluator:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    def _normalize(self, text: str) -> str:
        if not self.normalize:
            return text.strip()
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

        reference = self._annotation(ref_segments, "speaker_id")
        hypothesis = self._annotation(prediction.get("diarization", []), "speaker")
        details = self.der_metric.compute_components(reference, hypothesis)
        der = self.der_metric.compute_metric(details)
        word_ops = jiwer.process_words(ref_text, hyp_text)
        char_ops = jiwer.process_characters(ref_text, hyp_text)
        def counts(ops):
            return {
                "hits": int(ops.hits), "substitutions": int(ops.substitutions),
                "deletions": int(ops.deletions), "insertions": int(ops.insertions),
                "reference_units": int(ops.hits + ops.substitutions + ops.deletions),
            }
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

    @staticmethod
    def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute corpus metrics from total errors, not averaged call percentages."""
        if not metrics:
            return {"wer": None, "cer": None, "der": None, "rtf": None}

        def pooled_edits(key: str) -> tuple[float | None, dict[str, int]]:
            keys = ("hits", "substitutions", "deletions", "insertions", "reference_units")
            total = {k: 0 for k in keys}
            for row in metrics:
                counts = row.get(key)
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

    def __init__(self, transcriber: Any, evaluator: CallEvaluator | None = None,
                 audio_loader: AppTekAudioLoader | None = None):
        self.transcriber = transcriber
        self.evaluator = evaluator or CallEvaluator(normalize=True)
        self.audio_loader = audio_loader or AppTekAudioLoader()

    @staticmethod
    def save_excel(rows: list[dict], output_dir: Path) -> None:
        import xlsxwriter

        columns = [
            ("Call ID", 27), ("Audio Link", 18), ("WhisperX Transcript", 65),
            ("Reference Transcript", 65), ("Short Summary", 50),
            ("Customer Issue", 45), ("Agent Action", 45), ("Outcome / Next Step", 45),
            ("Overall Sentiment", 19), ("Beginning Sentiment", 20), ("End Sentiment", 19),
            ("Sentiment Reason", 52), ("WER", 12), ("CER", 12), ("DER", 12),
            ("RTF", 12), ("Duration (s)", 15), ("Processing (s)", 16),
            ("Domain", 21), ("Accent", 16), ("Analysis Status", 25),
        ]
        destination = output_dir / "call_evaluation.xlsx"
        temporary = output_dir / "call_evaluation.tmp.xlsx"
        with xlsxwriter.Workbook(str(temporary), {"strings_to_formulas": False, "strings_to_urls": False}) as workbook:
            sheet = workbook.add_worksheet("Call Results")
            header = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#213955", "valign": "vcenter"})
            text_format = workbook.add_format({"text_wrap": True, "valign": "top"})
            link_format = workbook.add_format({"font_color": "blue", "underline": True})
            number_format = workbook.add_format({"num_format": "0.00", "valign": "top"})
            percent_format = workbook.add_format({"num_format": "0.00%", "valign": "top"})
            for col, (title, width) in enumerate(columns):
                sheet.set_column(col, col, width)
                sheet.write(0, col, title, header)
            sheet.set_row(0, 28)
            sheet.freeze_panes(1, 2)
            sheet.autofilter(0, 0, max(len(rows), 1), len(columns) - 1)
            for index, record in enumerate(rows, start=1):
                sheet.set_row(index, 60)
                for col, (title, _) in enumerate(columns):
                    value = record.get(title, "")
                    if title == "Audio Link" and value:
                        sheet.write_url(index, col, value, link_format, "Listen")
                    elif isinstance(value, (int, float)):
                        sheet.write_number(index, col, value, percent_format if title in ("WER", "CER", "DER") else number_format)
                    else:
                        sheet.write_string(index, col, str(value or "")[:32767], text_format)
        temporary.replace(destination)

    @staticmethod
    def transcript_text(segments: list[dict], speaker_key: str) -> str:
        return "\n".join(
            f"[{float(s.get('start', 0)):.1f}-{float(s.get('end', 0)):.1f}s] "
            f"{s.get(speaker_key) or 'UNKNOWN'}: {s.get('text', '').strip()}"
            for s in segments
        )

    def run(self, limit: int = 10, output_dir: str | Path = "results",
            rows: Any = None, reuse_predictions: bool = True) -> dict[str, Any]:
        from src.llm import CallLLM

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        metrics = []
        failures = []
        excel_rows = []
        source = rows if rows is not None else AppTekDataset.stream(limit=limit)
        llm = CallLLM()

        for i, row in enumerate(source, start=1):
            if i > limit:
                break
            file_name = str(row.get("file_name") or f"call_{i:03d}.wav")
            output_path = target / f"{i:03d}_{Path(file_name).stem}.json"
            analysis_path = target / "summaries" / f"{output_path.stem}_summary.json"
            try:
                if reuse_predictions and output_path.exists():
                    prediction = json.loads(output_path.read_text(encoding="utf-8"))
                    print(f"[{i}/{limit}] Reusing {output_path.name}")
                else:
                    audio_path = self.audio_loader.load(row)
                    prediction = self.transcriber.transcribe(audio_path, output_path)

                score = self.evaluator.evaluate(prediction, row)
                metrics.append(score)

                ref_path = target / "reference_transcripts" / f"{Path(file_name).stem}.json"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                ref_path.write_text(json.dumps({
                    "call_id": Path(file_name).stem, "file_name": file_name,
                    "segments": [{"start": s.get("start"), "end": s.get("end"),
                                  "speaker": s.get("speaker_id"), "text": s.get("text", "")}
                                 for s in row.get("segments", [])],
                }, ensure_ascii=False, indent=2), encoding="utf-8")

                analysis = {}
                status = "Complete"
                try:
                    analysis = llm.analyze(prediction, analysis_path, reuse=reuse_predictions)
                except Exception as error:
                    status = f"LLM pending: {type(error).__name__}"
                    print(f"[{i}/{limit}] {status}: {error}")

                summary = analysis.get("summary") or {}
                sentiment = analysis.get("sentiment") or {}
                excel_rows.append({
                    "Call ID": prediction.get("call_id") or output_path.stem,
                    "Audio Link": self.audio_loader.audio_url(row),
                    "WhisperX Transcript": self.transcript_text(prediction["segments"], "speaker"),
                    "Reference Transcript": self.transcript_text(row["segments"], "speaker_id"),
                    "Short Summary": summary.get("short_summary"),
                    "Customer Issue": summary.get("customer_issue"),
                    "Agent Action": summary.get("agent_action"),
                    "Outcome / Next Step": summary.get("final_outcome_or_next_step"),
                    "Overall Sentiment": sentiment.get("label"),
                    "Beginning Sentiment": sentiment.get("beginning"),
                    "End Sentiment": sentiment.get("end"),
                    "Sentiment Reason": sentiment.get("justification"),
                    "WER": score["wer"], "CER": score["cer"], "DER": score["der"],
                    "RTF": score["rtf"], "Duration (s)": score["duration_seconds"],
                    "Processing (s)": score["processing_seconds"],
                    "Domain": score["domain"], "Accent": score["accent"],
                    "Analysis Status": status,
                })
                self.save_excel(excel_rows, target)
                print(f"[{i}/{limit}] WER={score['wer']:.3f} CER={score['cer']:.3f} DER={score['der']:.3f} | Excel saved")
            except Exception as error:
                failures.append({"call_index": i, "file_name": file_name, "error": str(error)})
                print(f"[{i}/{limit}] FAILED {file_name}: {error}")

        self.evaluator.save_metrics(metrics, target)
        (target / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        summary = {
            "requested_calls": limit,
            "successful_calls": len(metrics),
            "failed_calls": len(failures),
            "aggregation": "pooled rates are primary; macro mean/std retained for diagnostics",
            "overall": self.evaluator.aggregate(metrics),
            "mean": {}, "std": {},
        }
        for name in ("wer", "cer", "der", "rtf"):
            values = [float(m[name]) for m in metrics if m.get(name) is not None]
            summary["mean"][name] = statistics.mean(values) if values else None
            summary["std"][name] = statistics.stdev(values) if len(values) > 1 else None
        (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.save_excel(excel_rows, target)
        print(f"Excel: {target / 'call_evaluation.xlsx'} ({len(excel_rows)} calls)")
        return {"summary": summary, "metrics": metrics, "failures": failures}
