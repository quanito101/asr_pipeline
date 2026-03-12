#!/usr/bin/env python3
"""
Stage 3 – Add Gaussian noise at multiple SNR levels to each utterance.

Reads:  data/manifests/{lang}/clean.jsonl
Writes: data/manifests/{lang}/noisy_snr{SNR}.jsonl  (one per SNR level)
        data/noisy/{lang}/snr{SNR}/{stem}.wav

Usage:
    python src/add_noise.py --lang fr

Requirements:
    pip install numpy soundfile pyyaml tqdm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Core noise functions (from lab handout)                                      #
# --------------------------------------------------------------------------- #

def add_noise(signal: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def add_noise_to_file(input_wav: str, output_wav: str,
                      snr_db: float, seed: int | None = None) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError(f"Only mono audio is supported, got shape {signal.shape}")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)


# --------------------------------------------------------------------------- #
# SNR tag helper                                                               #
# --------------------------------------------------------------------------- #

def snr_tag(snr_db: int) -> str:
    """Convert SNR value to a safe filename tag, e.g. -5 -> 'snrm5', 10 -> 'snrp10'."""
    sign = "m" if snr_db < 0 else "p"
    return f"snr{sign}{abs(snr_db)}"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'fr'")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    lang = args.lang
    snr_levels = params["snr_levels"]
    seed = params["seed"]

    clean_manifest = Path("data/manifests") / lang / "clean.jsonl"
    if not clean_manifest.exists():
        print(f"ERROR: Clean manifest not found: {clean_manifest}")
        print("  Run download_data.py and phonemize.py first.")
        sys.exit(1)

    with open(clean_manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    print(f"[add_noise] {len(entries)} utterances × {len(snr_levels)} SNR levels "
          f"= {len(entries) * len(snr_levels)} noisy files to generate")

    manifest_dir = Path("data/manifests") / lang

    for snr_db in snr_levels:
        tag = snr_tag(snr_db)
        noisy_wav_dir = Path("data/noisy") / lang / tag
        noisy_wav_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = manifest_dir / f"noisy_{tag}.jsonl.tmp"
        final_path = manifest_dir / f"noisy_{tag}.jsonl"

        with open(tmp_path, "w", encoding="utf-8") as out:
            for entry in tqdm(entries, desc=f"[{lang}] SNR={snr_db:+d} dB"):
                stem = Path(entry["wav_path"]).stem
                noisy_wav = noisy_wav_dir / f"{stem}.wav"

                # Deterministic per-utterance seed for reproducibility
                utt_seed = (seed ^ abs(hash(entry["utt_id"]))) & 0xFFFFFFFF

                add_noise_to_file(
                    input_wav=entry["wav_path"],
                    output_wav=str(noisy_wav),
                    snr_db=snr_db,
                    seed=utt_seed,
                )

                noisy_entry = dict(entry)
                noisy_entry["wav_path"] = str(noisy_wav).replace("\\", "/")
                noisy_entry["snr_db"] = snr_db
                out.write(json.dumps(noisy_entry, ensure_ascii=False) + "\n")

        # Atomic replace (works on Windows unlike rename)
        tmp_path.replace(final_path)
        print(f"  → {final_path.name}")

    print(f"\n[{lang}] Done. Noisy manifests written to {manifest_dir}")


if __name__ == "__main__":
    main()
