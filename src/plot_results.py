#!/usr/bin/env python3
"""
Stage 6 – Plot PER vs SNR per language and cross-language mean.

Reads:  data/metrics/{lang}/per.json  (for each language in params.yaml)
Writes: data/figures/per_vs_snr.png

Usage:
    python src/plot_results.py

Requirements:
    pip install matplotlib pyyaml
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml


def load_metrics(lang: str) -> dict:
    path = Path("data/metrics") / lang / "per.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}. Run evaluate.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    languages = params["languages"]
    figures_dir = Path("data/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    all_snrs = None
    all_per_curves = []

    for lang in languages:
        data = load_metrics(lang)
        results = data["results"]

        # Separate clean baseline from noisy results
        clean = next((r for r in results if r["snr_db"] is None), None)
        noisy = sorted(
            [r for r in results if r["snr_db"] is not None],
            key=lambda r: r["snr_db"]
        )

        snrs = [r["snr_db"] for r in noisy]
        pers = [r["per"] for r in noisy]

        # Plot noisy curve
        ax.plot(snrs, pers, marker="o", linewidth=2, label=lang)

        # Plot clean baseline as a dashed horizontal line
        if clean:
            ax.axhline(
                clean["per"], linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"{lang} clean ({clean['per']:.3f})"
            )

        if all_snrs is None:
            all_snrs = snrs
        all_per_curves.append(pers)

    # Cross-language mean (only meaningful with > 1 language)
    if len(all_per_curves) > 1:
        mean_pers = np.mean(all_per_curves, axis=0).tolist()
        ax.plot(
            all_snrs, mean_pers,
            marker="s", linewidth=2.5, linestyle="-.",
            color="black", label="mean"
        )

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("PER", fontsize=13)
    ax.set_title("Phoneme Error Rate vs Noise Level", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    # Noisier (lower SNR) on the left
    ax.invert_xaxis()

    out_path = figures_dir / "per_vs_snr.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


if __name__ == "__main__":
    main()
