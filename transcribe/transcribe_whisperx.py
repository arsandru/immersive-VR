#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Iterable, List


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / 'data' / 'audio'
DEFAULT_OUTPUT = BASE_DIR / 'data' / 'transcripts'


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
}


def detect_default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def detect_default_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    if device == "mps":
        return "float32"
    return "int8"


def parse_args() -> argparse.Namespace:
    default_device = detect_default_device()
    default_compute_type = detect_default_compute_type(default_device)

    parser = argparse.ArgumentParser(
        description="Transcribe and diarize recordings with WhisperX."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Audio/video file or directory containing recordings. Defaults to data/audio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where transcripts and diarization outputs are written. Defaults to data/transcripts.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="WhisperX model name. Example: large-v3, medium, small.",
    )
    parser.add_argument(
        "--language",
        default="pt",
        help="Language code for transcription/alignment. Defaults to Portuguese (pt).",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device to use, e.g. cuda, cpu, mps. Defaults to an automatic selection for the current machine.",
    )
    parser.add_argument(
        "--compute-type",
        default=default_compute_type,
        help="WhisperX compute type, e.g. float16, int8, float32. Defaults to a device-appropriate selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for WhisperX transcription.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="Hugging Face token for diarization models. Defaults to HF_TOKEN/HUGGINGFACE_TOKEN env var.",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable diarization even if a Hugging Face token is available.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--include-audio-ext",
        action="store_true",
        help="Keep the original audio extension in output file stems.",
    )
    return parser.parse_args()


def iter_audio_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
            yield candidate


def safe_stem(path: Path, include_audio_ext: bool) -> str:
    if include_audio_ext:
        return path.name.replace(".", "_")
    return path.stem


def write_text_outputs(
    output_base: Path,
    result: dict,
    diarized: bool,
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_base.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    txt_path = output_base.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as f:
        for seg in result.get("segments", []):
            speaker = seg.get("speaker")
            text = seg.get("text", "").strip()
            if speaker:
                f.write(f"[{speaker}] {text}\n")
            else:
                f.write(f"{text}\n")

    srt_path = output_base.with_suffix(".srt")
    with srt_path.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(result.get("segments", []), start=1):
            f.write(f"{idx}\n")
            f.write(
                f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
            )
            speaker = seg.get("speaker")
            text = seg.get("text", "").strip()
            if speaker:
                f.write(f"[{speaker}] {text}\n\n")
            else:
                f.write(f"{text}\n\n")

    if diarized:
        diarization_segments = [
            {
                "speaker": seg.get("speaker"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
            for seg in result.get("segments", [])
        ]
        diar_path = output_base.with_name(output_base.name + "_diarization.json")
        with diar_path.open("w", encoding="utf-8") as f:
            json.dump(diarization_segments, f, ensure_ascii=False, indent=2)


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def release_model(*objs: object) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    try:
        import whisperx
    except ImportError as exc:
        raise SystemExit(
            "WhisperX is not installed. Install it first, for example:\n"
            "pip install whisperx"
        ) from exc

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    files = list(iter_audio_files(input_path))
    if not files:
        raise SystemExit(f"No supported audio/video files found in: {input_path}")

    diarization_enabled = not args.no_diarization and bool(args.hf_token)
    if not diarization_enabled and not args.no_diarization:
        print(
            "No Hugging Face token provided. Proceeding without diarization. "
            "Set HF_TOKEN or pass --hf-token to enable speaker diarization."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} file(s) to process.")
    print(f"Output directory: {output_dir}")
    print(f"Diarization enabled: {diarization_enabled}")

    model = whisperx.load_model(
        args.model,
        args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    align_model = None
    align_metadata = None
    align_language = None
    diarize_model = None
    if diarization_enabled:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=args.hf_token,
            device=args.device,
        )

    try:
        for index, audio_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] Processing {audio_path.name}")
            audio = whisperx.load_audio(str(audio_path))
            transcription = model.transcribe(audio, batch_size=args.batch_size)

            language_code = transcription.get("language")
            if not language_code:
                language_code = args.language or "en"

            if align_model is None or align_language != language_code:
                release_model(align_model)
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=args.device,
                )
                align_language = language_code

            aligned = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                args.device,
                return_char_alignments=False,
            )

            diarized = False
            final_result = aligned
            if diarize_model is not None:
                diar_segments = diarize_model(
                    audio,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                final_result = whisperx.assign_word_speakers(diar_segments, aligned)
                diarized = True

            stem = safe_stem(audio_path, include_audio_ext=args.include_audio_ext)
            write_text_outputs(output_dir / stem, final_result, diarized=diarized)
    finally:
        release_model(model, align_model, diarize_model)


if __name__ == "__main__":
    main()
