import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from sklearn.decomposition import PCA


EMBEDDING_NAME = "glove-wiki-gigaword-300"
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {"a", "an", "the", "and", "at"}
TOKEN_RE = re.compile(r"[A-Za-z']+")

try:
    import gensim.downloader as api
except ImportError as exc:
    raise RuntimeError(
        "This script requires gensim. Install it with `pip install gensim`."
    ) from exc

print(f"Loading {EMBEDDING_NAME} ...")
model = api.load(EMBEDDING_NAME)
embedding_cache = {}


def normalize_text(w: str) -> str:
    if not isinstance(w, str):
        return str(w)

    key = w.strip().lower()
    w = w.replace("_", " ")
    if key == "tune_out":
        return "tune out"
    if key in {"bit_worried", "bit worried"}:
        return "a bit worried"
    if key in {"bit_bothered", "bit bothered"}:
        return "a bit bothered"
    if key in {"like", "liked"}:
        return "liked it"
    return w


def tokenize(text: str):
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    return content_tokens if content_tokens else tokens


def average_token_vectors(tokens):
    vecs = [model[token] for token in tokens if token in model]
    if not vecs:
        return None
    return np.mean(np.vstack(vecs), axis=0)


def encode_word(w: str) -> np.ndarray:
    normalized = normalize_text(w)
    if normalized in embedding_cache:
        return embedding_cache[normalized]

    tokens = tokenize(normalized)
    vec = average_token_vectors(tokens)
    if vec is None:
        raise ValueError(f"No GloVe vectors found for: {w!r} -> {normalized!r}")

    embedding_cache[normalized] = np.asarray(vec, dtype=float)
    return embedding_cache[normalized]


def get_semantic_subspace(pos_words, neg_words, encode_fn):
    pos_vecs = np.vstack([encode_fn(w) for w in pos_words])
    neg_vecs = np.vstack([encode_fn(w) for w in neg_words])

    a_vec = pos_vecs.mean(axis=0)
    b_vec = neg_vecs.mean(axis=0)
    diff_mat = np.array([p - n for p in pos_vecs for n in neg_vecs])
    diff_vec = diff_mat.mean(axis=0)
    return a_vec, b_vec, diff_vec


distress_words = [
    "distressed", "danger", "apprehensive", "anguish", "torment", "agitated"
]
relaxed_words = [
    "relieved", "comforted", "security", "soothed", "reassured", "at ease"
]

a_vec, b_vec, distress_relaxed_axis = get_semantic_subspace(
    distress_words,
    relaxed_words,
    encode_word
)

axis_vec = distress_relaxed_axis
axis_norm = np.linalg.norm(axis_vec)

probe_words = [
    "worried",
    "little worried",
    "a little worried",
    "slightly worried",
    "a bit worried",
    "somewhat worried",
    "mildly worried",
    "anxious",
    "less anxious",
    "slightly anxious",
    "somewhat anxious",
    "a bit anxious",
    "some anxiety",
    "a little bothered",
    "very nervous",
    "extremely nervous",
    "calm",
    "relaxed",
    "engaged",
    "immersed",
    "distracted"
]


def semantic_projection(word, axis_vec, axis_norm, encode_fn):
    v = encode_fn(word)
    if axis_norm == 0:
        return np.nan
    return float(axis_vec @ v / axis_norm)


print("\nProbe word projections on the distress-relaxed axis:")
for probe_word in probe_words:
    try:
        probe_score = semantic_projection(probe_word, axis_vec, axis_norm, encode_word)
        print(f"{probe_word:15s} {probe_score: .4f}")
    except ValueError as exc:
        print(f"{probe_word:15s} OOV ({exc})")


def remove_not_words(cell):
    if isinstance(cell, list):
        return [w for w in cell if isinstance(w, str) and not w.startswith("not_")]
    if isinstance(cell, str):
        try:
            words = ast.literal_eval(cell)
            if isinstance(words, list):
                return [w for w in words if isinstance(w, str) and not w.startswith("not_")]
        except Exception:
            return []
    return []


def process_trial_single_axis(word_list, axis_vec, axis_norm, encode_fn):
    word_projs = []
    for w in word_list:
        if not isinstance(w, str):
            continue
        try:
            proj = semantic_projection(w, axis_vec, axis_norm, encode_fn)
        except ValueError:
            continue
        word_projs.append({"word": w, "projection": proj})

    if not word_projs:
        return {"word_projections": [], "mean_projection": np.nan}

    return {
        "word_projections": word_projs,
        "mean_projection": float(np.mean([wp["projection"] for wp in word_projs]))
    }


df = pd.read_csv(DATA_DIR / "emotion_words_checked.csv")
df_t = pd.read_csv(DATA_DIR / "Flow_current.csv")

df["ID"] = df["ID"].astype(str)
df_t["ID"] = df_t["ID"].astype(str)
df_t.columns = df_t.columns.str.strip()
condition_col = "Exp_Condition" if "Exp_Condition" in df_t.columns else "Exp_Condition "
merged = df.merge(df_t[["ID", condition_col]], on="ID", how="left")
merged["text"] = merged["text"].apply(remove_not_words)

word_level_rows = []
participant_level_rows = []

for _, row in merged.iterrows():
    participant_id = row["ID"]
    condition = row[condition_col]
    word_list = row["text"]

    if not word_list:
        continue

    result = process_trial_single_axis(word_list, axis_vec, axis_norm, encode_word)
    if not result["word_projections"]:
        continue

    for wp in result["word_projections"]:
        word_level_rows.append({
            "Participant": participant_id,
            "condition": condition,
            "word": wp["word"],
            "projection_fear_vs_calm": wp["projection"]
        })

    participant_level_rows.append({
        "Participant": participant_id,
        "condition": condition,
        "mean_projection_fear_vs_calm": result["mean_projection"],
        "word_list": str(word_list)
    })

word_level_df = pd.DataFrame(word_level_rows).dropna(subset=["condition"])
participant_level_df = pd.DataFrame(participant_level_rows).dropna(subset=["condition"])
word_level_df.to_csv(OUTPUT_DIR / "semantic_projection_glove300.csv", index=False)
participant_level_df.to_csv(OUTPUT_DIR / "semantic_projection_glove300_mean.csv", index=False)

data_for_plot = merged.dropna(subset=[condition_col]).copy()

all_words = []
all_vecs = []
all_conditions = []

for _, row in data_for_plot.iterrows():
    condition = row[condition_col]
    word_list = row["text"]
    if not word_list:
        continue
    for w in word_list:
        if not isinstance(w, str):
            continue
        try:
            vec = encode_word(w)
        except ValueError:
            continue
        all_words.append(w)
        all_vecs.append(vec)
        all_conditions.append(condition)

if not all_vecs:
    raise ValueError("No words with GloVe embeddings found for plotting.")

all_vecs = np.vstack(all_vecs)
axis_unit = axis_vec / axis_norm
x_coord = (all_vecs @ axis_vec) / axis_norm
proj_along_unit = all_vecs @ axis_unit
residuals = all_vecs - np.outer(proj_along_unit, axis_unit)
y_coord = PCA(n_components=1).fit_transform(residuals).ravel()

plot_df = pd.DataFrame({
    "word": all_words,
    "condition": all_conditions,
    "x": x_coord,
    "y": y_coord
})

plot_df = (
    plot_df.groupby(["word", "condition"], as_index=False)
    .agg(
        x=("x", "mean"),
        y=("y", "mean"),
        frequency=("word", "size")
    )
)

freq_min = plot_df["frequency"].min()
freq_max = plot_df["frequency"].max()
if freq_max == freq_min:
    plot_df["point_size"] = 80
    plot_df["label_size"] = 8
    plot_df["alpha"] = 0.75
else:
    freq_scaled = (plot_df["frequency"] - freq_min) / (freq_max - freq_min)
    plot_df["point_size"] = 80 + 220 * freq_scaled
    plot_df["label_size"] = 7 + 4 * freq_scaled
    plot_df["alpha"] = 0.55 + 0.35 * freq_scaled

unique_conditions = sorted(plot_df["condition"].unique())
condition_angles = np.linspace(0, 2 * np.pi, num=len(unique_conditions), endpoint=False)
condition_offsets = {
    cond: (0.12 * np.cos(angle), 0.12 * np.sin(angle))
    for cond, angle in zip(unique_conditions, condition_angles)
}
cond_colors = {
    "1": "#8de5a1",
    "2": "#ffb482",
    "3": "#a1c9f4"
}
plot_df["x_plot"] = plot_df["x"]
plot_df["y_plot"] = plot_df.apply(lambda row: row["y"] + condition_offsets[row["condition"]][1], axis=1)
plot_df.to_csv(OUTPUT_DIR / "semantic_projection_glove300_plot_points.csv", index=False)

plt.figure(figsize=(10, 8))
for cond in unique_conditions:
    cond_df = plot_df[plot_df["condition"] == cond]
    plt.scatter(
        cond_df["x_plot"],
        cond_df["y_plot"],
        s=cond_df["point_size"],
        color=cond_colors.get(str(cond), "#999999"),
        alpha=cond_df["alpha"],
        label=str(cond),
        edgecolor="white",
        linewidth=0.4
    )

label_df = (
    plot_df.groupby("word", as_index=False)
    .agg(
        x_plot=("x_plot", "mean"),
        y_plot=("y_plot", "mean"),
        label_size=("label_size", "max"),
        alpha=("alpha", "max")
    )
)
label_df.to_csv(OUTPUT_DIR / "semantic_projection_glove300_plot_labels.csv", index=False)

texts = []
for _, row in label_df.iterrows():
    texts.append(
        plt.text(
            row["x_plot"] + 0.01,
            row["y_plot"] + 0.01,
            row["word"],
            fontsize=row["label_size"],
            alpha=row["alpha"]
        )
    )

adjust_text(
    texts,
    x=label_df["x_plot"].to_numpy(),
    y=label_df["y_plot"].to_numpy(),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
    expand_points=(1.1, 1.2),
    expand_text=(1.05, 1.2)
)

plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
plt.title("Semantic Projection of Emotional Words on Distress-Relaxed Axis (GloVe 300d)", fontsize=18)
plt.xlabel("Projection on distress_vs_relaxed axis  ( (u·v) / ||u|| )", fontsize=17)
plt.ylabel("Orthogonal semantic variation (PC1 of residuals)", fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title="Condition", title_fontsize=15, fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "semantic_projection_glove300.pdf", format="pdf")
plt.savefig(OUTPUT_DIR / "semantic_projection_glove300.svg", format="svg")
plt.show()
