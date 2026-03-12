#!/usr/bin/env python3
"""
Stage 2 – Phonemize the ref_text field in a clean manifest using espeak-ng,
and write the result back as an updated manifest with ref_phon filled in.

Reads:  data/manifests/{lang}/clean.jsonl
Writes: data/manifests/{lang}/clean.jsonl  (updated in-place, atomically)

Usage:
    python src/phonemize.py --lang fr

Requirements:
    System: espeak-ng must be installed and on PATH
    pip install pyyaml tqdm
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


# Map our short lang codes to espeak-ng voice names
ESPEAK_LANG_MAP = {
    "fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "nl": "nl",
    "pt": "pt",
    "pl": "pl",
    "en": "en-us",
}


def check_espeak():
    """Verify espeak-ng is available on PATH."""
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, text=True
        )
        print(f"[phonemize] Found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: espeak-ng not found on PATH.")
        print("  Install from: https://github.com/espeak-ng/espeak-ng/releases")
        print("  Then add 'C:\\Program Files\\eSpeak NG' to your system PATH.")
        sys.exit(1)


def phonemize(text: str, espeak_lang: str) -> str:
    """
    Run espeak-ng on a single text string and return the IPA phoneme sequence.
    Uses subprocess so it works on any OS with espeak-ng installed.
    """
    result = subprocess.run(
        ["espeak-ng", "-v", espeak_lang, "-q", "--ipa", text],
        capture_output=True, text=True, encoding="utf-8"
    )
    # Strip leading underscore, whitespace, and trailing newlines
    phonemes = result.stdout.strip().lstrip("_").strip()
    # Collapse multiple spaces/newlines into single space
    phonemes = " ".join(phonemes.split())
    return phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'fr'")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    lang = args.lang

    if lang not in ESPEAK_LANG_MAP:
        print(f"ERROR: lang='{lang}' not in ESPEAK_LANG_MAP.")
        print(f"  Supported: {list(ESPEAK_LANG_MAP.keys())}")
        sys.exit(1)

    espeak_lang = ESPEAK_LANG_MAP[lang]

    manifest_path = Path("data/manifests") / lang / "clean.jsonl"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        print("  Run download_data.py first.")
        sys.exit(1)

    check_espeak()

    # Read existing manifest
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    print(f"[phonemize] Processing {len(entries)} utterances for lang={lang}...")

    # Write updated manifest atomically via temp file
    tmp_path = manifest_path.with_suffix(".jsonl.tmp")
    failed = 0

    with open(tmp_path, "w", encoding="utf-8") as out:
        for entry in tqdm(entries, desc=f"[{lang}] phonemizing"):
            text = entry.get("ref_text", "")
            try:
                ref_phon = phonemize(text, espeak_lang)
            except Exception as e:
                print(f"\n  WARNING: espeak failed for: '{text}' — {e}")
                ref_phon = ""
                failed += 1

            entry["ref_phon"] = ref_phon
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Atomic rename (replace works on Windows unlike rename)
    tmp_path.replace(manifest_path)

    print(f"[{lang}] Done. {len(entries) - failed}/{len(entries)} utterances phonemized.")
    print(f"[{lang}] Manifest updated: {manifest_path}")

    # Show a sample
    with open(manifest_path, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print(f"\nSample:")
    print(f"  ref_text: {sample['ref_text']}")
    print(f"  ref_phon: {sample['ref_phon']}")


if __name__ == "__main__":
    main()
