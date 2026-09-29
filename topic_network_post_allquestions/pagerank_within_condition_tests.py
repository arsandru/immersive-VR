#!/usr/bin/env python3
"""Within-condition tests for the PageRank analysis.

For every condition and flow scope:

1. Which topic is most central (highest PageRank) and is it more central than
   chance? Null = order shuffled inside every sequence (topic frequencies and
   sequence lengths kept, transition structure destroyed). The top topic is
   compared with the null distribution of the *maximum* PageRank.
2. Which topics are more/less central than their frequency predicts? Two-sided
   permutation test per topic against the same null, Benjamini-Hochberg over topics.
3. Which topics contribute significantly to the most central topic? PageRank
   satisfies PR_c = (1-a)/K + a * sum_j PR_j * P_jc (dangling topics spread
   uniformly), so each source j contributes a*PR_j*P_jc. Only sources with an
   observed transition into c are tested (one-sided against the shuffled-order
   null, BH over those sources); the rest of PR_c is reported as "baseline"
   (teleport + dangling redistribution). Participant-bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_network_analysis import (  # noqa: E402
    CONDITIONS, DEFAULT_INPUT, DEFAULT_OUTPUT, SCOPES, bh_fdr, build_participants,
    load_data,
)

ALPHA = 0.85


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def transition_matrix(C: np.ndarray) -> np.ndarray:
    K = len(C)
    rows = C.sum(axis=1)
    P = np.full((K, K), 1 / K)  # dangling topics jump uniformly
    nz = rows > 0
    P[nz] = C[nz] / rows[nz, None]
    return P


def pagerank(P: np.ndarray) -> np.ndarray:
    K = len(P)
    return np.linalg.solve(np.eye(K) - ALPHA * P.T, np.full(K, (1 - ALPHA) / K))


def flatten(pids, people, scope):
    """Concatenate all sequences; return topic array and segment ids."""
    arrs, segs, s = [], [], 0
    for pid in pids:
        for seq in people[pid][scope]:
            arrs.append(np.asarray(seq))
            segs.append(np.full(len(seq), s))
            s += 1
    return np.concatenate(arrs), np.concatenate(segs)


def counts_from(arr, seg, K):
    a, b = arr[:-1], arr[1:]
    m = (seg[:-1] == seg[1:]) & (a != b)
    C = np.zeros((K, K))
    np.add.at(C, (a[m], b[m]), 1)
    return C


def shuffled(arr, seg, rng):
    order = np.lexsort((rng.random(len(arr)), seg))
    return arr[order]


def stats(C):
    P = transition_matrix(C)
    pr = pagerank(P)
    return pr, ALPHA * pr[:, None] * P  # contrib[j, c]


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    df = load_data(args.input, include_outlier=False)
    topics = sorted(df["Topic"].unique().tolist())
    K = len(topics)
    people = build_participants(df, np.zeros((len(df), 1)), topics)
    by_cond = {c: [p for p, v in people.items() if v["condition"] == c] for c in CONDITIONS}
    from topic_network_analysis import labels_for  # noqa: E402
    labels = labels_for(topics)

    topic_rows, contrib_rows = [], []
    for scope in SCOPES:
        for cond in CONDITIONS:
            pids = by_cond[cond]
            arr, seg = flatten(pids, people, scope)
            C = counts_from(arr, seg, K)
            pr, contrib = stats(C)
            top = int(np.argmax(pr))

            # --- permutation null (order shuffled within sequences)
            null_pr = np.zeros((args.n_perm, K))
            null_contrib = np.zeros((args.n_perm, K))
            for i in range(args.n_perm):
                pr_n, con_n = stats(counts_from(shuffled(arr, seg, rng), seg, K))
                null_pr[i] = pr_n
                null_contrib[i] = con_n[:, top]

            # --- participant bootstrap
            boot_pr = np.zeros((args.n_boot, K))
            boot_contrib = np.zeros((args.n_boot, K))
            for i in range(args.n_boot):
                sample = list(rng.choice(pids, size=len(pids), replace=True))
                a, s_ = flatten(sample, people, scope)
                pr_b, con_b = stats(counts_from(a, s_, K))
                boot_pr[i] = pr_b
                boot_contrib[i] = con_b[:, top]
            pr_lo, pr_hi = np.percentile(boot_pr, [2.5, 97.5], axis=0)
            co_lo, co_hi = np.percentile(boot_contrib, [2.5, 97.5], axis=0)

            # (1)+(2) topic-level PageRank tests
            p_two = np.array([
                (1 + np.sum(np.abs(null_pr[:, t] - null_pr[:, t].mean())
                            >= abs(pr[t] - null_pr[:, t].mean()) - 1e-12)) / (1 + args.n_perm)
                for t in range(K)])
            p_fdr = bh_fdr(p_two)
            p_max = (1 + np.sum(null_pr.max(axis=1) >= pr[top] - 1e-12)) / (1 + args.n_perm)
            for t in range(K):
                topic_rows.append({
                    "scope": scope, "condition": cond, "Topic": topics[t],
                    "topic_label": labels[topics[t]], "pagerank": pr[t],
                    "ci_low": pr_lo[t], "ci_high": pr_hi[t],
                    "null_mean": null_pr[:, t].mean(),
                    "null_low": np.percentile(null_pr[:, t], 2.5),
                    "null_high": np.percentile(null_pr[:, t], 97.5),
                    "p_perm_two_sided": p_two[t], "p_fdr": p_fdr[t],
                    "is_top": t == top,
                    "p_top_vs_null_max": p_max if t == top else np.nan})

            # (3) contributions to the most central topic
            p_one = np.array([
                (1 + np.sum(null_contrib[:, j] >= contrib[j, top] - 1e-12)) / (1 + args.n_perm)
                for j in range(K)])
            tested = C[:, top] > 0  # only sources with an observed transition into the target
            p_one = np.where(tested, p_one, np.nan)
            p_one_fdr = np.full(K, np.nan)
            p_one_fdr[tested] = bh_fdr(p_one[tested])
            edge_total = contrib[tested, top].sum()
            for j in range(K):
                contrib_rows.append({
                    "scope": scope, "condition": cond,
                    "target_topic": topics[top], "target_label": labels[topics[top]],
                    "target_pagerank": pr[top], "source_topic": topics[j],
                    "source_label": labels[topics[j]], "n_transitions_to_target": C[j, top],
                    "source_pagerank": pr[j], "contribution": contrib[j, top],
                    "share_of_target_pagerank": contrib[j, top] / pr[top],
                    "ci_low": co_lo[j], "ci_high": co_hi[j],
                    "null_mean": null_contrib[:, j].mean(),
                    "p_perm_one_sided": p_one[j], "p_fdr": p_one_fdr[j]})
            contrib_rows.append({
                "scope": scope, "condition": cond, "target_topic": topics[top],
                "target_label": labels[topics[top]], "target_pagerank": pr[top],
                "source_topic": -99, "source_label": "Baseline (teleport + dangling)",
                "n_transitions_to_target": np.nan, "source_pagerank": np.nan,
                "contribution": pr[top] - edge_total,
                "share_of_target_pagerank": (pr[top] - edge_total) / pr[top]})
            print(f"{scope:17s} {cond:8s} top={labels[topics[top]]:22s} "
                  f"PR={pr[top]:.3f}  p(top vs null max)={p_max:.3f}", flush=True)

    pd.DataFrame(topic_rows).to_csv(out / "pagerank_within_condition_topic_tests.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(out / "pagerank_top_topic_contributions.csv", index=False)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
