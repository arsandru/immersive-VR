"""Human-reviewed labels for the broad shared sentence-level BERTopic model."""

TOPIC_LABELS = {
    -1: "Unassigned or mixed content",
    0: "Calm and relaxation",
    1: "Compassionate and professional care",
    2: "Subjective emotional and experiential effects",
    3: "Surgical concerns and coping",
    4: "Movement and sense of place",
    5: "Nature imagery and sensory responses",
    6: "Waiting experience",
    7: "Positive appraisal and optimism",
    8: "No perceived change and brief appraisals",
    9: "Anxiety and nervousness",
    10: "Feeling well and stable",
    11: "Feeling welcomed",
}


def topic_label(topic: int) -> str:
    return TOPIC_LABELS.get(int(topic), f"Topic {int(topic)}")


# Compact labels for figures where the full labels above are too long.
SHORT_LABELS = {
    -1: "Unassigned",
    0: "Calm",
    1: "Caring staff",
    2: "Personal effects",
    3: "Surgery worries",
    4: "Place & movement",
    5: "Nature & senses",
    6: "Waiting",
    7: "Optimism",
    8: "No change",
    9: "Anxiety",
    10: "Feeling well",
    11: "Welcomed",
}


def short_topic_label(topic: int) -> str:
    return SHORT_LABELS.get(int(topic), f"Topic {int(topic)}")
