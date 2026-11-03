"""Topic labels of the post-interview topic model: single source of truth is pre_post/timepoint_topic_labels.py, shared with
pre_post/ and topic_network_post/ so that clusters, labels and parameters are identical everywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pre_post"))
from timepoint_topic_labels import TOPIC_LABELS_POST as TOPIC_LABELS  # noqa: E402


def topic_label(topic: int) -> str:
    return TOPIC_LABELS.get(int(topic), f"Topic {int(topic)}")


short_topic_label = topic_label
