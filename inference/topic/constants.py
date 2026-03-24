"""Constants and paths for topic modeling."""

import os

TOPIC_MODELS_DIR = "artifacts/bertopic_by_cluster"
ALL_TOPIC_MODEL_PATH = "artifacts/bertopic_all_n30.model"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ALL_EMBEDDING_MODEL_PATH = "artifacts/review_embedder_v1"

CLUSTER_TO_MODEL = {
    "Cluster 1 (AI-Charged Product/Service Providers)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_1.model"
    ),
    "Cluster 2 (AI Development Facilitators)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_2.model"
    ),
    "Cluster 3 (Data Analytics Providers)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_3.model"
    ),
    "Cluster 4 (Deep Tech Researchers)": os.path.join(
        TOPIC_MODELS_DIR, "bertopic_cluster_4.model"
    ),
}
