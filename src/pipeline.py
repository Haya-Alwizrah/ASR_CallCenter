from __future__ import annotations
import gc
import json
import time
from pathlib import Path
import torch
import whisperx

class CallCenterWhisperX:
    def __init__(self, hf_token: str, model_name: str = "small", language: str | None = "en", device: str | None = None, compute_type: str | None = None, batch_size: int = 4, num_speakers: int | None = 2, keep_asr_loaded: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        self.num_speakers = num_speakers
        self.keep_asr_loaded = keep_asr_loaded
        self.hf_token = hf_token
        self._asr = None
        self._diarizer = None
        self._aligner = None
        self._align_metadata = None
        self._align_language = None
        self._align_device = "cpu" if self.device == "cuda" else self.device

    def _cleanup(self) -> None:
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _load_asr(self):
        if self._asr is None:
            self._asr = whisperx.load_model(
                self.model_name, self.device,
                compute_type=self.compute_type, language=self.language
            )
        return self._asr

    def load_models(self):
        self._load_asr()
        self._load_aligner(self.language)
        self._load_diarizer()
        return self

    def _load_aligner(self, language: str):
        if self._aligner is None or self._align_language != language:
            self._aligner, self._align_metadata = whisperx.load_align_model(language_code=language, device=self._align_device)
            self._align_language = language
        return self._aligner, self._align_metadata

    def _load_diarizer(self):
        if self._diarizer is None:
            from whisperx.diarize import DiarizationPipeline
            self._diarizer = DiarizationPipeline(token=self.hf_token, device=self.device)
        return self._diarizer

    def unload_asr(self) -> None:
        self._asr = None
        self._cleanup()

    def transcribe(self, audio_path, output_path=None):
        path = Path(audio_path)
        
        t0 = time.perf_counter()
        audio = whisperx.load_audio(str(path))
        duration = len(audio) / 16000.0  # whisperx.load_audio returns 16-kHz mono

        asr_result = self._load_asr().transcribe(audio, batch_size=self.batch_size)

        if not self.keep_asr_loaded:
            self.unload_asr()

        language = asr_result.get("language") or self.language
        
        align_model, metadata = self._load_aligner(language)
        aligned = whisperx.align(
            asr_result["segments"], align_model, metadata, audio,
            self._align_device, return_char_alignments=False
        )

        kwargs = {"num_speakers": self.num_speakers} if self.num_speakers is not None else {}
        diarized_df = self._diarizer(audio, **kwargs)
        assigned = whisperx.assign_word_speakers(diarized_df, aligned)

        # Keep raw diarization intervals for accurate DER evaluation.
        diarization = []
        for _, row in diarized_df.iterrows():
            diarization.append({
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "speaker": str(row["speaker"]),
            })

        segments = []
        for seg in assigned["segments"]:
            words = []
            for word in seg.get("words", []):
                item = {"text": str(word.get("word", "")), "speaker": word.get("speaker")}

                if word.get("start") is not None:
                    item["start"] = round(float(word["start"]), 3)
                if word.get("end") is not None:
                    item["end"] = round(float(word["end"]), 3)
                if word.get("score") is not None:
                    item["score"] = round(float(word["score"]), 4)
                words.append(item)

            segments.append({
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": str(seg.get("text", "")).strip(),
                "words": words,
            })

        answer = {
            "call_id": path.stem,
            "source_file": path.name,
            "model": self.model_name,
            "language": language,
            "duration_seconds": round(duration, 3),
            "processing_seconds": round(time.perf_counter() - t0, 3),
            "segments": segments,
            "diarization": diarization,
            "notes": [
                "Speaker IDs are model clusters, not verified Customer/Agent roles.",
                "Some words may have no aligned timestamps or no assigned speaker.",
            ],
        }
        
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
        return answer

    def close(self):
        self._diarizer = None
        self._aligner = None
        self._align_metadata = None
        self._align_language = None
        self.unload_asr()