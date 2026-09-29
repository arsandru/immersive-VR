#!/usr/bin/env python3
"""How distant are the pre topics from the post topics?

Topics are extracted separately per timepoint (fit_timepoint_topics.py: UMAP + HDBSCAN on that timepoint's descriptors only).
Here the two topic sets are compared directly in the embedding space, with no need for a shared topic model.

Topic = unit-length mean embedding of its member descriptors (unassigned descriptors excluded). D[i, j] = 1 - cos(pre topic i,
post topic j).
  * per topic: nearest counterpart in the other timepoint and its distance (a topic far from every topic of the other
    timepoint is "new" or "gone");
  * set distance S = size-weighted mean nearest-counterpart distance, averaged over both directions (0 = every topic has an
    identical counterpart); M = mean distance under the optimal one-to-one matching of the topics (Hungarian);
  * per condition: A_c = mean distance from each POST descriptor of the condition to its nearest PRE topic, B_c = the reverse
    (how far the condition's content lies from the other timepoint's topic structure).
Reference for "no time effect": pre/post labels are swapped at random within each participant observed at both timepoints (whole
descriptor lists move together), both topic models are refitted on the pseudo-timepoints and every statistic is recomputed
(``--n-perm`` refits). p = share of null values >= observed (large distance = big change). Between conditions: A_c and B_c
differences with condition labels permuted across participants (topics fixed). BH-FDR within families (three conditions; three pairs).
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
from prepost_common import CONDITIONS, OUT, bh_fdr, label_time  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--drop-review", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_labels(emb, mcs):
    from hdbscan import HDBSCAN
    from umap import UMAP
    r = UMAP(n_neighbors=min(10, len(emb) - 1), n_components=5, min_dist=0.0, metric="cosine", random_state=42).fit_transform(emb)
    return HDBSCAN(min_cluster_size=mcs, min_samples=2, metric="euclidean", cluster_selection_method="eom").fit_predict(r)


def centroids(emb, lab):
    ids = [t for t in sorted(set(lab)) if t >= 0]
    C = np.stack([emb[lab == t].mean(axis=0) for t in ids])
    return ids, C / np.linalg.norm(C, axis=1, keepdims=True), np.array([(lab == t).sum() for t in ids], float)


def cond_set_distance(emb_pre, lab_pre, cond_pre, emb_post, lab_post, cond_post, c, min_n=2):
    """S restricted to one condition: the condition's own descriptors define each topic's centre (topics fixed);
    topics with fewer than ``min_n`` descriptors of the condition are left out."""
    def cents(emb, lab, cond):
        ids = [t for t in sorted(set(lab)) if t >= 0 and ((lab == t) & (cond == c)).sum() >= min_n]
        if not ids:
            return None, None
        C = np.stack([emb[(lab == t) & (cond == c)].mean(axis=0) for t in ids])
        return C / np.linalg.norm(C, axis=1, keepdims=True), np.array([((lab == t) & (cond == c)).sum() for t in ids], float)
    Cp, sp = cents(emb_pre, lab_pre, cond_pre)
    Cq, sq = cents(emb_post, lab_post, cond_post)
    if Cp is None or Cq is None:
        return np.nan
    D = 1 - Cp @ Cq.T
    return 0.5 * ((sp / sp.sum()) @ D.min(axis=1) + (sq / sq.sum()) @ D.min(axis=0))


def stats(emb_pre, lab_pre, emb_post, lab_post, cond_pre, cond_post):
    ip, Cp, sp = centroids(emb_pre, lab_pre)
    iq, Cq, sq = centroids(emb_post, lab_post)
    D = 1 - Cp @ Cq.T
    near_pre = D.min(axis=1)   # for each pre topic: distance to the closest post topic
    near_post = D.min(axis=0)
    S = 0.5 * ((sp / sp.sum()) @ near_pre + (sq / sq.sum()) @ near_post)
    r, c = linear_sum_assignment(D)
    M = D[r, c].mean()
    # descriptor -> nearest topic of the other timepoint
    d_post_to_pre = (1 - emb_post @ Cp.T).min(axis=1)
    d_pre_to_post = (1 - emb_pre @ Cq.T).min(axis=1)
    A = {c_: d_post_to_pre[cond_post == c_].mean() for c_ in range(3)}
    B = {c_: d_pre_to_post[cond_pre == c_].mean() for c_ in range(3)}
    Sc = {c_: cond_set_distance(emb_pre, lab_pre, cond_pre, emb_post, lab_post, cond_post, c_) for c_ in range(3)}
    return {"Sc": Sc, "D": D, "ip": ip, "iq": iq, "sp": sp, "sq": sq, "near_pre": near_pre, "near_post": near_post, "S": S, "M": M,
            "A": A, "B": B, "d_post_to_pre": d_post_to_pre, "d_pre_to_post": d_pre_to_post}


def main() -> None:
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    pre = pd.read_csv(OUT / "pre_units_with_topics.csv")
    post = pd.read_csv(OUT / "post_units_with_topics.csv")
    if a.drop_review:
        keep = set(pre.participant_id[~pre.needs_review]) & set(post.participant_id[~post.needs_review])
        pre = pre[~pre.needs_review]
        post = post[~post.needs_review]
    z = np.load(OUT / "unit_embeddings.npz")
    lk = dict(zip(z["document_id"].tolist(), z["emb"]))

    def emb_of(u):
        e = np.stack([lk[i] for i in u["document_id"]])
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    Epre, Epost = emb_of(pre), emb_of(post)
    cpre = pre.condition_label.map(CONDITIONS.index).to_numpy()
    cpost = post.condition_label.map(CONDITIONS.index).to_numpy()
    lab_pre = fit_labels(Epre, a.min_cluster_size) if a.drop_review else pre.Topic.to_numpy()
    lab_post = fit_labels(Epost, a.min_cluster_size) if a.drop_review else post.Topic.to_numpy()
    obs = stats(Epre, lab_pre, Epost, lab_post, cpre, cpost)

    # ---- pooled unit table for the swaps
    allu = pd.concat([pre.assign(t=0), post.assign(t=1)], ignore_index=True)
    Eall = np.vstack([Epre, Epost])
    call = allu.condition_label.map(CONDITIONS.index).to_numpy()
    pid = allu.participant_id.to_numpy()
    paired = sorted(set(pre.participant_id) & set(post.participant_id))
    t0 = allu.t.to_numpy()

    null = {"S": [], "M": [], "near": [], **{f"A{c}": [] for c in range(3)}, **{f"B{c}": [] for c in range(3)}, **{f"S{c}": [] for c in range(3)}}
    for _ in range(a.n_perm):
        flip = set(p for p in paired if rng.random() < 0.5)
        t_new = np.where(np.isin(pid, list(flip)), 1 - t0, t0)
        m0, m1 = t_new == 0, t_new == 1
        l0, l1 = fit_labels(Eall[m0], a.min_cluster_size), fit_labels(Eall[m1], a.min_cluster_size)
        if (l0 >= 0).sum() == 0 or (l1 >= 0).sum() == 0 or len(set(l0[l0 >= 0])) < 2 or len(set(l1[l1 >= 0])) < 2:
            continue
        s = stats(Eall[m0], l0, Eall[m1], l1, call[m0], call[m1])
        null["S"].append(s["S"])
        null["M"].append(s["M"])
        null["near"].extend(list(s["near_pre"]) + list(s["near_post"]))
        for c in range(3):
            null[f"A{c}"].append(s["A"][c])
            null[f"B{c}"].append(s["B"][c])
            null[f"S{c}"].append(s["Sc"][c])
    nnull = len(null["S"])
    pv = lambda o, n: (1 + np.sum(np.asarray(n) >= o - 1e-12)) / (1 + len(n))  # noqa: E731

    # ---- per-topic table
    thr = float(np.percentile(null["near"], 95))
    rows = []
    for side, ids, sizes, near, D, other_ids, other_t, this_t in (
            ("pre", obs["ip"], obs["sp"], obs["near_pre"], obs["D"], obs["iq"], "post", "pre"),
            ("post", obs["iq"], obs["sq"], obs["near_post"], obs["D"].T, obs["ip"], "pre", "post")):
        for k, t in enumerate(ids):
            j = int(np.argmin(D[k]))
            rows.append({"timepoint": this_t, "topic": t, "label": label_time(this_t, t), "n_descriptors": int(sizes[k]),
                         "nearest_topic_other_timepoint": label_time(other_t, other_ids[j]), "distance": near[k],
                         "null_95th_percentile": thr, "far_from_every_topic_of_other_timepoint": bool(near[k] > thr)})
    pd.DataFrame(rows).to_csv(OUT / f"topic_distance_per_topic{a.tag}.csv", index=False)
    pd.DataFrame(obs["D"], index=[label_time("pre", t) for t in obs["ip"]],
                 columns=[label_time("post", t) for t in obs["iq"]]).to_csv(OUT / f"topic_distance_matrix{a.tag}.csv")

    # ---- summary tests
    summ = [{"analysis": "set distance S (size-weighted nearest counterpart)", "comparison": "all participants", "statistic": obs["S"],
             "null_mean": np.mean(null["S"]), "null_95": np.percentile(null["S"], 95), "p_perm": pv(obs["S"], null["S"])},
            {"analysis": "set distance M (optimal one-to-one matching)", "comparison": "all participants", "statistic": obs["M"],
             "null_mean": np.mean(null["M"]), "null_95": np.percentile(null["M"], 95), "p_perm": pv(obs["M"], null["M"])}]
    cond_rows = []
    names = {"A": "post descriptors -> nearest pre topic (A)", "B": "pre descriptors -> nearest post topic (B)",
             "S": "set distance S within the condition (condition's own topic centres)"}
    for c, cname in enumerate(CONDITIONS):
        for key, val in (("S", obs["Sc"][c]), ("A", obs["A"][c]), ("B", obs["B"][c])):
            n = np.asarray(null[f"{key}{c}"])
            n = n[~np.isnan(n)]
            cond_rows.append({"analysis": names[key],
                              "comparison": cname, "statistic": val, "null_mean": np.mean(n), "null_95": np.percentile(n, 95),
                              "p_perm": pv(val, n)})
    res = pd.DataFrame(summ + cond_rows)
    res["p_fdr"] = np.nan
    for an in res.analysis.unique():
        m = res.analysis == an
        if m.sum() == 3:
            res.loc[m, "p_fdr"] = bh_fdr(res.loc[m, "p_perm"].to_numpy())

    # ---- between conditions (labels permuted across participants, topics fixed)
    pair_rows = []
    pids_pre = pre.participant_id.to_numpy()
    pids_post = post.participant_id.to_numpy()
    cond_of = {**dict(zip(pids_pre, cpre)), **dict(zip(pids_post, cpost))}
    pu = np.array(sorted(cond_of))
    cond0 = np.array([cond_of[p] for p in pu])
    pos = {p: i for i, p in enumerate(pu)}
    ipre = np.array([pos[p] for p in pids_pre])
    ipost = np.array([pos[p] for p in pids_post])

    def AB(lab):
        A = [obs["d_post_to_pre"][lab[ipost] == c].mean() for c in range(3)]
        B = [obs["d_pre_to_post"][lab[ipre] == c].mean() for c in range(3)]
        return np.array(A), np.array(B)

    def SC(lab):
        return np.array([cond_set_distance(Epre, lab_pre, lab[ipre], Epost, lab_post, lab[ipost], c) for c in range(3)])

    oA, oB = AB(cond0)
    oS = SC(cond0)
    nA, nB, nS = [], [], []
    for _ in range(a.n_perm):
        perm = rng.permutation(cond0)
        x, y = AB(perm)
        nA.append(x)
        nB.append(y)
        nS.append(SC(perm))
    nA, nB, nS = np.array(nA), np.array(nB), np.array(nS)
    for key, o, n_ in (("S: set distance within the condition", oS, nS), ("A: post descriptors -> nearest pre topic", oA, nA),
                       ("B: pre descriptors -> nearest post topic", oB, nB)):
        fam = []
        for i, j in combinations(range(3), 2):
            d = o[i] - o[j]
            fam.append({"analysis": "between conditions, " + key, "comparison": f"{CONDITIONS[i]} vs {CONDITIONS[j]}",
                        "statistic": d, "null_mean": float(np.mean(n_[:, i] - n_[:, j])),
                        "null_95": float(np.percentile(np.abs(n_[:, i] - n_[:, j]), 95)),
                        "p_perm": (1 + np.sum(np.abs(n_[:, i] - n_[:, j]) >= abs(d) - 1e-12)) / (1 + len(n_))})
        f = pd.DataFrame(fam)
        f["p_fdr"] = bh_fdr(f.p_perm.to_numpy())
        pair_rows.append(f)
    res = pd.concat([res] + pair_rows, ignore_index=True)
    res.to_csv(OUT / f"topic_distance_tests{a.tag}.csv", index=False)
    np.savez(OUT / f"topic_distance_null{a.tag}.npz", S=null["S"], M=null["M"], near=null["near"],
             **{k: null[k] for k in null if k[0] in "ABS" and len(k) == 2})
    (OUT / f"topic_distance_audit{a.tag}.json").write_text(json.dumps(
        {"n_perm_requested": a.n_perm, "n_perm_valid": nnull, "n_pre_topics": len(obs["ip"]), "n_post_topics": len(obs["iq"]),
         "null_far_threshold_95th": thr}, indent=1))
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(res.round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
