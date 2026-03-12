#!/usr/bin/env python3
"""
Stage 5 – Compute Phoneme Error Rate (PER) for each prediction manifest.

PER = (S + D + I) / N  — normalised character-level edit distance on IPA.
Both ref_phon and hyp_phon are normalised to bare IPA characters (stress
marks and spaces stripped) before comparison, making this equivalent to CER
as suggested in the lab handout.

Reads:  data/predictions/{lang}/*.jsonl
Writes: data/metrics/{lang}/per.json

Usage:
    python src/evaluate.py --lang fr

Requirements:
    pip install pyyaml tqdm
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import yaml
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# PER computation                                                              #
# --------------------------------------------------------------------------- #

def normalize_phon(s: str) -> list:
    """
    Strip stress marks, combining diacritics, spaces and punctuation,
    returning a list of bare IPA characters for edit-distance comparison.
    """
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        # Skip: spaces, modifier letters (stress ˈˌ), combining marks,
        # punctuation, hyphens, underscores
        if ch in (' ', '-', '_', '.', "'"):
            continue
        if cat in ('Lm', 'Mn', 'Po', 'Pd', 'Sk', 'Zs'):
            continue
        out.append(ch)
    return out


def token_error_rate(ref: list, hyp: list) -> float:
    """Standard DP edit distance normalised by len(ref)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m] / n


def compute_per(ref_phon: str, hyp_phon: str) -> float:
    ref_tokens = normalize_phon(ref_phon)
    hyp_tokens = normalize_phon(hyp_phon)
    if not ref_tokens:
        return 0.0
    return token_error_rate(ref_tokens, hyp_tokens)


# --------------------------------------------------------------------------- #
# Evaluate one manifest                                                        #
# --------------------------------------------------------------------------- #

def evaluate_manifest(pred_path: Path) -> dict:
    with open(pred_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    pers = []
    for e in entries:
        ref = e.get("ref_phon", "")
        hyp = e.get("hyp_phon", "")
        if ref.strip():
            pers.append(compute_per(ref, hyp))

    mean_per = sum(pers) / len(pers) if pers else float("nan")
    snr_db = entries[0].get("snr_db") if entries else None

    return {
        "snr_db": snr_db,
        "per": round(mean_per, 4),
        "n_utterances": len(pers),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'fr'")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    lang = args.lang
    pred_dir = Path("data/predictions") / lang
    metrics_dir = Path("data/metrics") / lang
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.jsonl"))
    if not pred_files:
        print(f"ERROR: No prediction manifests found in {pred_dir}")
        print("  Run run_inference.py first.")
        sys.exit(1)

    results = []
    for pf in tqdm(pred_files, desc=f"[{lang}] evaluating"):
        r = evaluate_manifest(pf)
        results.append(r)
        snr_str = f"{r['snr_db']:+d} dB" if r['snr_db'] is not None else "clean"
        print(f"  {pf.name:30s}  SNR={snr_str:>8s}  PER={r['per']:.4f}  (n={r['n_utterances']})")

    # Sort: clean first, then ascending SNR
    results.sort(key=lambda x: (x["snr_db"] is not None, x["snr_db"] or 0))

    tmp = metrics_dir / "per.json.tmp"
    final = metrics_dir / "per.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"lang": lang, "results": results}, f, indent=2)
    tmp.replace(final)

    print(f"\n[{lang}] Metrics written: {final}")


if __name__ == "__main__":
    main()
