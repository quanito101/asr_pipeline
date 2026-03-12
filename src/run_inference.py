#!/usr/bin/env python3
"""
Stage 4 – Run facebook/wav2vec2-lv-60-espeak-cv-ft on all manifests
(clean + noisy) for a given language, and write prediction manifests.

Reads:  data/manifests/{lang}/clean.jsonl
        data/manifests/{lang}/noisy_*.jsonl
Writes: data/predictions/{lang}/clean.jsonl
        data/predictions/{lang}/noisy_*.jsonl

Usage:
    python src/run_inference.py --lang fr

Requirements:
    pip install transformers torch soundfile tqdm pyyaml
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import yaml
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

TARGET_SR = 16000
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #

def load_model(model_name: str):
    print(f"[inference] Loading model: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[inference] Running on: {device}")
    if device == "cuda":
        print(f"[inference] GPU: {torch.cuda.get_device_name(0)}")
    return processor, model, device


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #

def transcribe_batch(wav_paths: list, processor, model, device) -> list:
    """Transcribe a batch of wav files, return list of phoneme strings."""
    signals = []
    for path in wav_paths:
        signal, sr = sf.read(path)
        if signal.ndim != 1:
            signal = signal.mean(axis=1)
        if sr != TARGET_SR:
            raise ValueError(
                f"Expected {TARGET_SR} Hz, got {sr} Hz in {path}. "
                "Re-run download_data.py to resample."
            )
        signals.append(signal)

    inputs = processor(
        signals,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    return transcriptions


def process_manifest(manifest_path: Path, out_path: Path,
                     processor, model, device, batch_size: int = 16) -> None:
    """Run inference on a manifest and write prediction manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as out:
        for i in tqdm(range(0, len(entries), batch_size),
                      desc=f"  {manifest_path.name}"):
            batch = entries[i: i + batch_size]
            wav_paths = [e["wav_path"] for e in batch]
            try:
                preds = transcribe_batch(wav_paths, processor, model, device)
            except Exception as exc:
                print(f"\n  WARNING: batch {i} failed ({exc}), using empty predictions")
                preds = [""] * len(batch)

            for entry, pred in zip(batch, preds):
                out_entry = dict(entry)
                out_entry["hyp_phon"] = pred.strip()
                out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

    # Atomic replace (works on Windows and Linux)
    tmp_path.replace(out_path)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'fr'")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    lang = args.lang
    manifest_dir = Path("data/manifests") / lang
    pred_dir = Path("data/predictions") / lang

    manifests = sorted(manifest_dir.glob("*.jsonl"))
    if not manifests:
        print(f"ERROR: No manifests found in {manifest_dir}")
        print("  Run download_data.py, phonemize.py and add_noise.py first.")
        sys.exit(1)

    processor, model, device = load_model(params["model"]["name"])

    for mpath in manifests:
        out_path = pred_dir / mpath.name
        if out_path.exists():
            print(f"[inference] Skipping (already exists): {out_path.name}")
            continue
        print(f"[inference] Processing: {mpath.name}")
        process_manifest(mpath, out_path, processor, model, device, args.batch_size)

    print(f"\n[{lang}] Inference complete. Predictions in: {pred_dir}")


if __name__ == "__main__":
    main()
