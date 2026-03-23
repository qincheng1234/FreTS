#!/usr/bin/env python3
"""evaluate_tar.py - Transient Assignment Responsiveness (TAR) evaluator.

Reads the routing_behavior_log.csv produced during the test phase and evaluates
whether the Wavelet expert is correctly dispatched to high-transient batches.

Metric definition (TAR - Transient Assignment Responsiveness):
  1. Identify high-transient batches: batch_crest_factor > crest_threshold (default 3.0).
     Crest Factor C_i = max(|Δx|) / σ(x) per batch.
  2. Compute mean Wavelet probability for those high-transient batches  (P_wav|C>3).
  3. Compute global mean Wavelet probability over ALL batches           (P_wav_global).
  4. Pass condition: P_wav|C>3  >=  shift_factor * P_wav_global (default shift_factor=1.5).
     i.e. the Wavelet expert assignment must surge by at least 50% during transient events.

Usage:
    python evaluate_tar.py --log_path results/<setting>/routing_behavior_log.csv
    python evaluate_tar.py --log_path results/<setting>/routing_behavior_log.csv \\
        --crest_threshold 3.0 --shift_factor 1.5
"""

import argparse
import sys

import pandas as pd


def evaluate_tar(
    log_path: str,
    crest_threshold: float = 3.0,
    shift_factor: float = 1.5,
) -> int:
    """Evaluate TAR metric from a routing behavior log CSV.

    Returns:
        0  - TAR criterion passed (routing is sensitive to transient events)
        1  - TAR criterion failed
        2  - Not enough data to evaluate (e.g. no high-transient batches found)
    """
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {log_path}", flush=True)
        return 1

    required_cols = {'batch_idx', 'batch_mse', 'prob_vmd', 'prob_wavelet', 'batch_crest_factor'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", flush=True)
        print(f"  Found columns: {list(df.columns)}", flush=True)
        return 1

    total_batches = len(df)
    if total_batches == 0:
        print("[ERROR] Log file is empty.", flush=True)
        return 1

    p_wav_global = df['prob_wavelet'].mean()
    high_transient = df[df['batch_crest_factor'] > crest_threshold]
    n_high = len(high_transient)

    print("=" * 60)
    print("  TAR (Transient Assignment Responsiveness) Report")
    print("=" * 60)
    print(f"  Log file          : {log_path}")
    print(f"  Total batches     : {total_batches}")
    print(f"  Crest threshold   : {crest_threshold}")
    print(f"  Required shift    : >= {shift_factor}x global mean")
    print(f"  High-transient    : {n_high} batches (crest > {crest_threshold})")
    print(f"  Global P_wavelet  : {p_wav_global:.4f}")
    print(f"  Global P_vmd      : {df['prob_vmd'].mean():.4f}")
    print(f"  Global MSE        : {df['batch_mse'].mean():.6f}")
    print("-" * 60)

    if n_high == 0:
        print("  [WARN] No high-transient batches found.")
        print("         Cannot evaluate TAR. The dataset may not contain strong")
        print("         transient signals, or crest_threshold is too high.")
        print("=" * 60)
        return 2

    p_wav_transient = high_transient['prob_wavelet'].mean()
    p_wav_ratio = p_wav_transient / (p_wav_global + 1e-8)
    mse_transient = high_transient['batch_mse'].mean()
    crest_mean = high_transient['batch_crest_factor'].mean()

    print(f"  High-transient P_wavelet : {p_wav_transient:.4f}")
    print(f"  Ratio (transient/global) : {p_wav_ratio:.3f}x")
    print(f"  High-transient MSE       : {mse_transient:.6f}")
    print(f"  High-transient avg crest : {crest_mean:.2f}")
    print("-" * 60)

    passed = p_wav_transient >= shift_factor * p_wav_global

    if passed:
        print(f"  RESULT: [PASS]")
        print(f"    Wavelet expert surged {p_wav_ratio:.2f}x (>= {shift_factor}x required).")
        print(f"    The dual-branch routing correctly dispatches transient signals.")
    else:
        print(f"  RESULT: [FAIL]")
        print(f"    Wavelet expert ratio {p_wav_ratio:.2f}x < {shift_factor}x required.")
        print(f"    The router is NOT sufficiently sensitive to transient events.")

    print("=" * 60)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transient Assignment Responsiveness (TAR) metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--log_path', type=str, required=True,
        help='Path to routing_behavior_log.csv generated during testing.',
    )
    parser.add_argument(
        '--crest_threshold', type=float, default=3.0,
        help='Crest factor threshold to identify high-transient batches (default: 3.0).',
    )
    parser.add_argument(
        '--shift_factor', type=float, default=1.5,
        help='Minimum required ratio of transient P_wavelet to global P_wavelet (default: 1.5).',
    )
    args = parser.parse_args()

    exit_code = evaluate_tar(
        log_path=args.log_path,
        crest_threshold=args.crest_threshold,
        shift_factor=args.shift_factor,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
