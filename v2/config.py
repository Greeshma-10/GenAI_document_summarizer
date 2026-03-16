import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ==========================
    # AWS / Region
    # ==========================
    AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")

    # ==========================
    # Model IDs
    # ==========================
    LLM_MODEL_ID: str = os.getenv(
        "LLM_MODEL_ID",
        "meta.llama3-8b-instruct-v1:0"
    )

    EMBED_MODEL_ID: str = os.getenv(
        "EMBED_MODEL_ID",
        "amazon.titan-embed-text-v2:0"
    )

    # ==========================
    # Generation Parameters
    # ==========================
    MAX_GEN_LEN_CHUNK: int = int(os.getenv("MAX_GEN_LEN_CHUNK", 350))
    MAX_GEN_LEN_SECTION: int = int(os.getenv("MAX_GEN_LEN_SECTION", 400))
    MAX_GEN_LEN_EXEC: int = int(os.getenv("MAX_GEN_LEN_EXEC", 700))

    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.0))
    TOP_P: float = float(os.getenv("TOP_P", 0.9))

    # ==========================
    # Retry Settings
    # ==========================
    MAX_RETRIES_LLM: int = int(os.getenv("MAX_RETRIES_LLM", 3))
    MAX_RETRIES_EMBED: int = int(os.getenv("MAX_RETRIES_EMBED", 2))
    BASE_DELAY: float = float(os.getenv("BASE_DELAY", 0.8))

    # ==========================
    # Clustering
    # ==========================
    BASE_DISTANCE_RESEARCH: float = float(
        os.getenv("BASE_DISTANCE_RESEARCH", 0.40)
    )

    BASE_DISTANCE_ACADEMIC: float = float(
        os.getenv("BASE_DISTANCE_ACADEMIC", 0.50)
    )

    MIN_CHUNKS_FOR_CLUSTERING: int = int(
        os.getenv("MIN_CHUNKS_FOR_CLUSTERING", 3)
    )

    # ==========================
    # Concurrency
    # ==========================
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))

    # ==========================
    # Logging
    # ==========================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
