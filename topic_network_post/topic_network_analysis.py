#!/usr/bin/env python3
"""Topic-network analysis with (probabilistic) semantic betweenness centrality.

Nodes are the sentence-level BERTopic topics from ``all_questions``. Two graphs
are built for every condition:

* flow graph      - directed; edge A->B when a sentence in topic A is followed by
                    a sentence in topic B. Two scopes: transitions inside one
                    response ("within_response"), or along a participant's whole
                    interview Q1->Q2->Q3 ("across_questions").
* semantic graph  - undirected k-nearest-neighbour graph on topic centroids of
                    Qwen3 sentence embeddings (edge length = 1 - cosine).

Centralities follow Lande et al., "Probabilistic Semantic Betweenness
Centrality in Cognitive Networks" (SBC / PSBC), normalised by (K-1)(K-2):

* SBC_hop    share of hop-shortest paths through a node (classical betweenness)
* PSBC_hop   the same paths, each weighted by its probability prod(P_ij)
* SBC_prob   flow graph only: optimal = most probable path (length -log P_ij)
* SBC_sim    semantic graph only: optimal = most similar path (length 1-cos)

"What is central to a condition" (as opposed to bridging) is measured with:

* PageRank        flow graphs: stationary weight of the transition graph
                  (damping 0.85, weights = P_ij); sums to 1, uniform = 1/K
* SemCentrality   how close a topic is, in embedding space, to everything else
                  the condition says: prevalence-weighted mean cosine similarity
                  to the other topics' centroids

Uncertainty: participant-level bootstrap. Condition differences: participant
level permutation test with Benjamini-Hochberg correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from all_questions.topic_labels import topic_label  # noqa: E402
from q3.q3_topic_analysis import EMBED_INSTRUCTION, MODEL_ID  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    BASE_DIR / "all_questions" / "outputs"
    / "combined_questions_sentence_topic_assignments.csv"
)
DEFAULT_OUTPUT = HERE / "outputs"
CONDITIONS = ["VR Art", "VR Only", "Control"]
SCOPES = ["within_response", "across_questions"]
# graph key -> measures computed on it
MEASURES = {
    "flow": ["SBC_hop", "PSBC_hop", "SBC_prob"],
    "semantic": ["SBC_hop", "PSBC_hop", "SBC_sim"],
}
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--knn", type=int, default=3,
                   help="neighbours per topic in the semantic graph")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--include-outlier", action="store_true",
                   help="keep topic -1 as a node (default: drop those sentences)")
    return p.parse_args()


# ----------------------------------------------------------------- data ----
def load_data(path: Path, include_outlier: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df["included_in_model"] & df["condition_label"].isin(CONDITIONS)]
    if not include_outlier:
        df = df.loc[df["Topic"] != -1]
    return df.sort_values(["participant_id", "question_id", "sentence_number"]).reset_index(drop=True)


def sentence_embeddings(df: pd.DataFrame, out: Path, args) -> np.ndarray:
    """Qwen3 embeddings of the sentences (same model/instruction as the topic model)."""
    cache = out / "sentence_embeddings.npz"
    ids = df["document_id"].to_numpy()
    if cache.exists():
        z = np.load(cache)
        lookup = dict(zip(z["document_id"].tolist(), z["emb"]))
        if all(i in lookup for i in ids):
            return np.stack([lookup[i] for i in ids])
    from sentence_transformers import SentenceTransformer

    kw = {"device": args.device} if args.device else {}
    model = SentenceTransformer(MODEL_ID, **kw)
    emb = model.encode([f"{EMBED_INSTRUCTION}{s}" for s in df["sentence"]],
                       batch_size=args.batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    np.savez(cache, document_id=ids, emb=emb)
    return emb


def build_participants(df: pd.DataFrame, emb: np.ndarray, topics: list[int]) -> dict:
    idx = {t: i for i, t in enumerate(topics)}
    people = {}
    for pid, g in df.groupby("participant_id", sort=True):
        rows = g.index.to_numpy()
        seq_all = [idx[t] for t in g["Topic"]]
        within = [[idx[t] for t in r["Topic"]] for _, r in g.groupby("response_id", sort=False)]
        people[pid] = {
            "condition": g["condition_label"].iloc[0],
            "within_response": within,
            "across_questions": [seq_all],
            "topic_idx": np.array(seq_all),
            "emb": emb[rows],
        }
    return people


# --------------------------------------------------------- graph building ---
def transition_counts(pids: list, people: dict, scope: str, K: int):
    C = np.zeros((K, K))
    self_moves = 0
    for pid in pids:
        for seq in people[pid][scope]:
            for a, b in zip(seq[:-1], seq[1:]):
                if a == b:
                    self_moves += 1
                else:
                    C[a, b] += 1
    return C, self_moves


def flow_graph(C: np.ndarray) -> nx.DiGraph:
    K = len(C)
    G = nx.DiGraph()
    G.add_nodes_from(range(K))
    rows = C.sum(axis=1)
    for i, j in zip(*np.nonzero(C)):
        p = C[i, j] / rows[i]
        G.add_edge(int(i), int(j), count=float(C[i, j]), prob=float(p),
                   length=float(-np.log(p)))
    return G


def semantic_inputs(pids: list, people: dict, K: int):
    counts = np.zeros(K)
    sums = None
    for pid in pids:
        p = people[pid]
        if sums is None:
            sums = np.zeros((K, p["emb"].shape[1]))
        np.add.at(sums, p["topic_idx"], p["emb"])
        np.add.at(counts, p["topic_idx"], 1)
    cent = np.zeros_like(sums)
    ok = counts > 0
    cent[ok] = sums[ok] / np.linalg.norm(sums[ok], axis=1, keepdims=True)
    return cent, counts


def semantic_graph(cent: np.ndarray, counts: np.ndarray, k: int) -> nx.DiGraph:
    """kNN graph (union of neighbour lists) on topics present in the sample.

    Edges are stored in both directions. ``prob`` follows the paper's
    weighting w_ij = Score(j) * sim_ij / d_in(j), Score = topic prevalence.
    """
    K = len(cent)
    valid = [i for i in range(K) if counts[i] > 0]
    S = cent @ cent.T
    und = set()
    for i in valid:
        others = sorted((j for j in valid if j != i), key=lambda j: -S[i, j])[:k]
        und.update((min(i, j), max(i, j)) for j in others)
    G = nx.DiGraph()
    G.add_nodes_from(range(K))
    for i, j in und:
        for a, b in ((i, j), (j, i)):
            G.add_edge(a, b, sim=float(S[a, b]), length=float(max(1 - S[a, b], EPS)))
    prev = counts / counts.sum()
    for a, b, d in G.edges(data=True):
        d["w"] = prev[b] * max(d["sim"], EPS) / max(G.in_degree(b), 1)
    for a in G.nodes:
        tot = sum(d["w"] for _, _, d in G.out_edges(a, data=True))
        for _, b, d in G.out_edges(a, data=True):
            d["prob"] = d["w"] / tot if tot > 0 else 0.0
    return G


# ---------------------------------------------------------- centralities ---
def hop_centralities(G: nx.DiGraph):
    """SBC_hop and PSBC_hop: enumerate all hop-shortest paths for every pair."""
    K = G.number_of_nodes()
    sbc = np.zeros(K)
    psbc = np.zeros(K)
    for s in G.nodes:
        for t in G.nodes:
            if s == t or not nx.has_path(G, s, t):
                continue
            paths = list(nx.all_shortest_paths(G, s, t))
            probs = [float(np.prod([G[a][b]["prob"] for a, b in zip(p[:-1], p[1:])]))
                     for p in paths]
            total = sum(probs)
            for p, pr in zip(paths, probs):
                for v in p[1:-1]:
                    sbc[v] += 1 / len(paths)
                    if total > 0:
                        psbc[v] += pr / total
    norm = (K - 1) * (K - 2)
    return sbc / norm, psbc / norm


def length_centrality(G: nx.DiGraph) -> np.ndarray:
    K = G.number_of_nodes()
    bc = nx.betweenness_centrality(G, weight="length", normalized=False)
    return np.array([bc[i] for i in range(K)]) / ((K - 1) * (K - 2))


def graph_measures(G: nx.DiGraph, kind: str) -> dict[str, np.ndarray]:
    sbc, psbc = hop_centralities(G)
    extra = "SBC_prob" if kind == "flow" else "SBC_sim"
    return {"SBC_hop": sbc, "PSBC_hop": psbc, extra: length_centrality(G)}


def pagerank_centrality(G: nx.DiGraph) -> np.ndarray:
    pr = nx.pagerank(G, alpha=0.85, weight="prob")
    return np.array([pr[i] for i in range(G.number_of_nodes())])


def semantic_centrality(cent: np.ndarray, counts: np.ndarray, gcent: np.ndarray) -> np.ndarray:
    """Prevalence-weighted mean cosine similarity of each topic to the other topics.

    Topic positions are the condition's own centroids; topics absent from the
    sample fall back to the all-participant centroid (they carry weight 0).
    """
    pos = np.where((counts > 0)[:, None], cent, gcent)
    share = counts / counts.sum()
    S = pos @ pos.T
    out = np.zeros(len(pos))
    for t in range(len(pos)):
        w = share.copy()
        w[t] = 0
        out[t] = (w * S[t]).sum() / w.sum() if w.sum() > 0 else 0
    return out


def all_measures(pids: list, people: dict, K: int, k: int, gcent: np.ndarray) -> dict:
    """{(graph, scope, measure): vector over topics} for one participant multiset."""
    out = {}
    for scope in SCOPES:
        C, _ = transition_counts(pids, people, scope, K)
        G = flow_graph(C)
        for m, v in graph_measures(G, "flow").items():
            out[("flow", scope, m)] = v
        out[("flow", scope, "PageRank")] = pagerank_centrality(G)
    cent, counts = semantic_inputs(pids, people, K)
    for m, v in graph_measures(semantic_graph(cent, counts, k), "semantic").items():
        out[("semantic", "centroids", m)] = v
    out[("semantic", "centroids", "SemCentrality")] = semantic_centrality(cent, counts, gcent)
    return out


# ------------------------------------------------------------- statistics ---
def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def labels_for(topics: list[int]) -> dict[int, str]:
    return {t: topic_label(t) for t in topics}


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_data(args.input, args.include_outlier)
    topics = sorted(df["Topic"].unique().tolist())
    K = len(topics)
    labels = {t: topic_label(t) for t in topics}
    emb = sentence_embeddings(df, out, args)
    people = build_participants(df, emb, topics)
    by_cond = {c: [p for p, v in people.items() if v["condition"] == c] for c in CONDITIONS}
    all_pids = [p for c in CONDITIONS for p in by_cond[c]]
    sizes = [len(by_cond[c]) for c in CONDITIONS]
    print(f"{len(df)} sentences, {len(people)} participants, {K} topics; sizes {dict(zip(CONDITIONS, sizes))}")

    gcent, _ = semantic_inputs(all_pids, people, K)

    # ---- observed graphs, edge tables and descriptives
    observed, edge_rows, desc_rows = {}, [], []
    for c in CONDITIONS:
        observed[c] = all_measures(by_cond[c], people, K, args.knn, gcent)
        for scope in SCOPES:
            C, self_moves = transition_counts(by_cond[c], people, scope, K)
            G = flow_graph(C)
            n_moves = C.sum() + self_moves
            desc_rows.append({"graph": "flow", "scope": scope, "condition": c,
                              "n_transitions": int(n_moves),
                              "self_transition_share": self_moves / n_moves if n_moves else np.nan,
                              "n_edges": G.number_of_edges(),
                              "density": nx.density(G)})
            for a, b, d in G.edges(data=True):
                edge_rows.append({"graph": "flow", "scope": scope, "condition": c,
                                  "source": topics[a], "target": topics[b],
                                  "source_label": labels[topics[a]], "target_label": labels[topics[b]],
                                  "count": d["count"], "probability": d["prob"]})
        cent, counts = semantic_inputs(by_cond[c], people, K)
        G = semantic_graph(cent, counts, args.knn)
        desc_rows.append({"graph": "semantic", "scope": "centroids", "condition": c,
                          "n_transitions": np.nan, "self_transition_share": np.nan,
                          "n_edges": G.number_of_edges() // 2, "density": nx.density(G)})
        for a, b, d in G.edges(data=True):
            if a < b:
                edge_rows.append({"graph": "semantic", "scope": "centroids", "condition": c,
                                  "source": topics[a], "target": topics[b],
                                  "source_label": labels[topics[a]], "target_label": labels[topics[b]],
                                  "count": np.nan, "probability": d["sim"]})
    # shared reference semantic graph (all participants)
    cent, counts = semantic_inputs(all_pids, people, K)
    np.save(out / "topic_centroids_all.npy", cent)
    pd.DataFrame(desc_rows).to_csv(out / "network_descriptives.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "network_edges.csv", index=False)

    # ---- bootstrap CIs
    keys = list(observed[CONDITIONS[0]].keys())
    boot = {k: np.zeros((args.n_boot, len(CONDITIONS), K)) for k in keys}
    for b in range(args.n_boot):
        for ci, c in enumerate(CONDITIONS):
            sample = list(rng.choice(by_cond[c], size=len(by_cond[c]), replace=True))
            res = all_measures(sample, people, K, args.knn, gcent)
            for k in keys:
                boot[k][b, ci] = res[k]
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", flush=True)

    rows = []
    for k in keys:
        for ci, c in enumerate(CONDITIONS):
            obs = observed[c][k]
            ranks = pd.Series(obs).rank(ascending=False, method="min").to_numpy()
            lo, hi = np.percentile(boot[k][:, ci], [2.5, 97.5], axis=0)
            for ti, t in enumerate(topics):
                rows.append({"graph": k[0], "scope": k[1], "measure": k[2], "condition": c,
                             "Topic": t, "topic_label": labels[t], "value": obs[ti],
                             "ci_low": lo[ti], "ci_high": hi[ti], "rank": int(ranks[ti])})
    pd.DataFrame(rows).to_csv(out / "centrality_by_condition.csv", index=False)

    # ---- permutation tests on condition differences
    pairs = list(combinations(range(len(CONDITIONS)), 2))
    obs_diff = {k: {(i, j): observed[CONDITIONS[i]][k] - observed[CONDITIONS[j]][k]
                    for i, j in pairs} for k in keys}
    exceed = {k: {pr: np.zeros(K) for pr in pairs} for k in keys}
    for n in range(args.n_perm):
        perm = list(rng.permutation(all_pids))
        groups, start = [], 0
        for s in sizes:
            groups.append(perm[start:start + s])
            start += s
        res = [all_measures(g, people, K, args.knn, gcent) for g in groups]
        for k in keys:
            for i, j in pairs:
                d = res[i][k] - res[j][k]
                exceed[k][(i, j)] += np.abs(d) >= np.abs(obs_diff[k][(i, j)]) - 1e-12
        if (n + 1) % 100 == 0:
            print(f"  permutation {n + 1}/{args.n_perm}", flush=True)
    trows = []
    for k in keys:
        for i, j in pairs:
            for ti, t in enumerate(topics):
                trows.append({"graph": k[0], "scope": k[1], "measure": k[2],
                              "condition_a": CONDITIONS[i], "condition_b": CONDITIONS[j],
                              "Topic": t, "topic_label": labels[t],
                              "value_a": observed[CONDITIONS[i]][k][ti],
                              "value_b": observed[CONDITIONS[j]][k][ti],
                              "difference": obs_diff[k][(i, j)][ti],
                              "p_perm": (1 + exceed[k][(i, j)][ti]) / (1 + args.n_perm)})
    tests = pd.DataFrame(trows)
    tests["p_fdr"] = tests.groupby(["graph", "scope", "measure"])["p_perm"].transform(
        lambda s: bh_fdr(s.to_numpy()))
    tests.to_csv(out / "centrality_condition_tests.csv", index=False)

    (out / "run_summary.json").write_text(json.dumps({
        "input": str(args.input), "n_sentences": int(len(df)), "n_participants": len(people),
        "participants_per_condition": dict(zip(CONDITIONS, sizes)), "topics": topics,
        "outlier_topic_included": args.include_outlier, "knn": args.knn,
        "n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed,
        "embedding_model": MODEL_ID}, indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
