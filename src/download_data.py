#!/usr/bin/env python3
"""
Stage 1 – Stream Multilingual LibriSpeech from HuggingFace and
produce a clean manifest: data/manifests/{lang}/clean.jsonl

Uses facebook/multilingual_librispeech which is:
  - Proper Parquet format (no loading script issues)
  - Freely streamable with no login
  - Multilingual: en, de, fr, es, it, nl, pt, pl
  - Already 16 kHz mono — no resampling needed

Usage:
    python src/download_data.py --lang en

Requirements:
    pip install datasets torchaudio soundfile pyyaml tqdm
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torchaudio
import torch
import yaml
from tqdm import tqdm

HF_DATASET = "facebook/multilingual_librispeech"

# Map our lang codes to MLS config names
LANG_MAP = {
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "nl": "dutch",
    "pt": "portuguese",
    "pl": "polish",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def audio_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_wav(audio_array, sr: int, dst: Path, target_sr: int = 16000) -> float:
    """Save numpy audio array as mono 16-kHz WAV. Returns duration in seconds."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    waveform = torch.tensor(audio_array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    torchaudio.save(str(dst), waveform, target_sr)
    return waveform.shape[1] / target_sr


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code: en, de, fr, es, it, nl, pt, pl")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    lang = args.lang
    max_utt = params["data"]["max_utterances"]
    target_sr = params["model"]["target_sr"]

    if lang not in LANG_MAP:
        print(f"ERROR: lang='{lang}' not supported.")
        print(f"  Supported: {list(LANG_MAP.keys())}")
        sys.exit(1)

    mls_config = LANG_MAP[lang]

    raw_dir = Path("data/raw") / lang / "wav"
    manifest_dir = Path("data/manifests") / lang
    manifest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"[download] Streaming {HF_DATASET}, config={mls_config}, split=test")
    print("[download] Only downloading the utterances we need.")

    ds = load_dataset(
        HF_DATASET,
        mls_config,
        split="test",
        streaming=True,
    )

    tmp_manifest = manifest_dir / "clean.jsonl.tmp"
    final_manifest = manifest_dir / "clean.jsonl"

    count = 0
    with open(tmp_manifest, "w", encoding="utf-8") as out:
        for sample in tqdm(ds, desc=f"[{lang}] downloading + saving wav", total=max_utt):
            if count >= max_utt:
                break

            # MLS fields: audio, file, text, speaker_id, chapter_id, id
            utt_id_raw = str(sample.get("id", f"{lang}_utt{count:06d}"))
            stem = utt_id_raw.replace("/", "_").replace("\\", "_")
            text = sample.get("transcript", sample.get("text", "")).strip()
            if not text:
                continue

            audio = sample["audio"]
            audio_array = audio["array"]
            sr = audio["sampling_rate"]

            wav_path = raw_dir / f"{stem}.wav"
            duration = save_wav(audio_array, sr, wav_path, target_sr)

            entry = {
                "utt_id": f"{lang}_{stem}",
                "lang": lang,
                "wav_path": str(wav_path).replace("\\", "/"),
                "ref_text": text,
                "ref_phon": "",          # filled by Stage 3
                "sr": target_sr,
                "duration_s": round(duration, 3),
                "snr_db": None,
                "audio_md5": audio_md5(str(wav_path)),
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        tmp_manifest.unlink(missing_ok=True)
        print("ERROR: No utterances saved. Check your internet connection.")
        sys.exit(1)

    tmp_manifest.rename(final_manifest)
    print(f"\n[{lang}] Done! Manifest: {final_manifest}  ({count} entries)")


if __name__ == "__main__":
    main()
