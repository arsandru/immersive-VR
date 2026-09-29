# Transcription Workflow

This directory contains the WhisperX-based transcription and diarization utility for local recordings.

## Script

- `transcribe/transcribe_whisperx.py`

By default it reads from:
- `data/audio/`

By default it writes transcripts to:
- `data/transcripts/`

That output directory is created when the script runs. It does not need to exist in the repository beforehand.

## Outputs

For each recording, the script writes:
- `.json`
- `.txt`
- `.srt`
- `_diarization.json` when diarization is enabled

## Requirements

Python packages:
- `whisperx`
- `torch`
- `faster-whisper`

System dependency:
- `ffmpeg`

On macOS with Apple Silicon, this script currently defaults to CPU because the local WhisperX/CTranslate2 setup may not support `mps`.

## Language and Diarization

- Default transcription language is Portuguese (`pt`).
- Diarization runs only when `HF_TOKEN` or `HUGGINGFACE_TOKEN` is available, unless disabled with `--no-diarization`.

## Example

Run all files in `data/audio/`:

```bash
source .venv/bin/activate
python transcribe/transcribe_whisperx.py
```

Run a single file:

```bash
source .venv/bin/activate
python transcribe/transcribe_whisperx.py data/audio/example.wav
```

Disable diarization:

```bash
python transcribe/transcribe_whisperx.py --no-diarization
```
