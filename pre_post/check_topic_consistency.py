#!/usr/bin/env python3
"""Check that the per-timepoint clusters, cluster labels and cluster parameters are identical in
pre_post/ (core networks), topic_network_pre/ and topic_network_post/.  Exit code 1 on any mismatch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


from timepoint_topic_labels import TOPIC_LABELS_POST, TOPIC_LABELS_PRE, TOPIC_MODEL_PARAMS  # noqa: E402

for t, labels in (("pre", TOPIC_LABELS_PRE), ("post", TOPIC_LABELS_POST)):
    src = pd.read_csv(HERE / "outputs" / f"{t}_units_with_topics.csv").set_index("document_id").Topic
    tn = pd.read_csv(ROOT / f"topic_network_{t}" / "outputs" / f"{t}_descriptor_topic_assignments.csv").set_index("document_id").Topic
    check(f"{t}: clusters in topic_network_{t} identical to pre_post/{t}_units_with_topics.csv", src.equals(tn), f"{len(tn)} descriptors")
    # labels: every module reads the same dict
    sys.path.insert(0, str(ROOT / f"topic_network_{t}"))
    mod = __import__(f"{t}_topic_labels")
    check(f"{t}: labels in topic_network_{t} identical to the shared source", mod.TOPIC_LABELS == labels)
    sys.path.pop(0)
    sys.modules.pop(f"{t}_topic_labels")
    import prepost_common as pc
    check(f"{t}: labels in pre_post/ identical to the shared source", pc.TIME_LABELS[t] == labels)
    # parameters recorded at fit time
    audit = json.load(open(HERE / "outputs" / f"{t}_topic_audit.json"))
    p = TOPIC_MODEL_PARAMS
    check(f"{t}: fit parameters equal the shared parameters", audit["min_cluster_size"] == p["min_cluster_size"]
          and audit["min_samples"] == p["min_samples"], f"min_cluster_size={audit['min_cluster_size']}, min_samples={audit['min_samples']}")
    check(f"{t}: every topic id has a label", set(int(x) for x in src.unique()) <= set(labels), f"topics {sorted(src.unique())}")

a, b = (json.load(open(HERE / "outputs" / f"{t}_topic_audit.json")) for t in ("pre", "post"))
check("pre and post fitted with the same parameters", (a["min_cluster_size"], a["min_samples"]) == (b["min_cluster_size"], b["min_samples"]))

# central topic per condition and timepoint agree between the core-network figure and the topic_network folders
sys.path.insert(0, str(HERE))
import plot_core_network as pcn  # noqa: E402
from prepost_common import label_time, load_units_time  # noqa: E402

u = pd.concat([load_units_time("pre"), load_units_time("post")], ignore_index=True)
for t in ("pre", "post"):
    tn = pd.read_csv(ROOT / f"topic_network_{t}" / "outputs" / "pagerank_within_condition_topic_tests.csv")
    tn = tn[(tn.scope == "within_response") & tn.is_top].set_index("condition").topic_label
    for c in pcn.ORDER:
        pr, C, n, nt = pcn.panel_data(u, c, t)
        K = pcn.K_OF[t]
        top = label_time(t, K[int(np.argmax(pr))])
        check(f"{t}/{c}: central topic in core_network_pre_vs_post equals topic_network_{t}", top == tn[c], f"{top} vs {tn[c]}")
print("ALL CONSISTENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
