"""Topic labels for the shared pre+post BERTopic model (pre_post/), used by the pre-interview topic-network analysis.
Short labels written after reading sample descriptors of every topic; topics 2, 8 and 9 overlap in meaning."""

TOPIC_LABELS = {
    -1: "Unassigned",
    0: "Anxiety & fear",
    1: "Feeling fine",
    2: "Calm & tranquil",
    3: "The surgery",
    4: "Mixed emotions",
    5: "Optimism",
    6: "Video & nature",
    7: "Expecting it to go well",
    8: "Feeling calmer",
    9: "Relaxed",
    10: "Wanting it over",
}


def topic_label(topic: int) -> str:
    return TOPIC_LABELS.get(int(topic), f"Topic {int(topic)}")


short_topic_label = topic_label
