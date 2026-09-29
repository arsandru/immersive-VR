"""Human-reviewed labels for the combined-response BERTopic model."""

RESPONSE_TOPIC_LABELS: dict[int, str] = {
    -1: "Unassigned or mixed content",
    0: "Feeling calmer and more relaxed",
    1: "Mixed emotional and sensory responses to VR",
    2: "Surgical anxiety and coping",
    3: "Positive appraisal and no complaints",
    4: "Waiting-related anxiety",
    5: "Immersive nature, serenity, and personal meaning",
    6: "Personalized and compassionate care",
    7: "Feeling warmly welcomed",
    8: "Feeling well and stable",
    9: "VR as a positive and enriching experience",
}


def response_topic_label(topic: int) -> str:
    return RESPONSE_TOPIC_LABELS.get(int(topic), f"Response topic {int(topic)}")


# Compact labels for figures (consistent with SHORT_LABELS in topic_labels.py).
SHORT_RESPONSE_LABELS: dict[int, str] = {
    -1: "Unassigned",
    0: "Calm",
    1: "Mixed VR reactions",
    2: "Surgery worries",
    3: "Positive, no complaints",
    4: "Waiting anxiety",
    5: "Nature & meaning",
    6: "Caring staff",
    7: "Welcomed",
    8: "Feeling well",
    9: "Enriching VR",
}


def short_response_topic_label(topic: int) -> str:
    return SHORT_RESPONSE_LABELS.get(int(topic), f"Response topic {int(topic)}")
