"""Topic inference package."""

from inference.topic.constants import (
    ALL_EMBEDDING_MODEL_PATH,
    ALL_TOPIC_MODEL_PATH,
    CLUSTER_TO_MODEL,
    EMBEDDING_MODEL_NAME,
    TOPIC_MODELS_DIR,
)
from inference.topic.discovery import discover_topics_batch
from inference.topic.loaders import load_topic_model, load_topic_model_all
from inference.topic.keywords import (
    extract_keywords_batch,
    extract_keywords_single,
)
from inference.topic.predict import predict_topic_batch, predict_topic_batch_all, predict_topic_single


def llm_label_topic(*args, **kwargs):
    from inference.topic.llm_label import llm_label_topic as _impl

    return _impl(*args, **kwargs)


def llm_label_topic_from_keywords(*args, **kwargs):
    from inference.topic.llm_label import llm_label_topic_from_keywords as _impl

    return _impl(*args, **kwargs)


def llm_label_topics_from_keywords(*args, **kwargs):
    from inference.topic.llm_label import llm_label_topics_from_keywords as _impl

    return _impl(*args, **kwargs)


def llm_topic_summary(*args, **kwargs):
    from inference.topic.llm_summary import llm_topic_summary as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "ALL_EMBEDDING_MODEL_PATH",
    "ALL_TOPIC_MODEL_PATH",
    "CLUSTER_TO_MODEL",
    "EMBEDDING_MODEL_NAME",
    "TOPIC_MODELS_DIR",
    "discover_topics_batch",
    "extract_keywords_batch",
    "extract_keywords_single",
    "llm_label_topic",
    "llm_label_topic_from_keywords",
    "llm_label_topics_from_keywords",
    "llm_topic_summary",
    "load_topic_model",
    "load_topic_model_all",
    "predict_topic_batch",
    "predict_topic_batch_all",
    "predict_topic_single",
]
