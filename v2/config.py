import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    # ── OCR / Tesseract ──────────────────────────────────────────────────────────
    TESSERACT_CMD: str = os.getenv(
        "TESSERACT_CMD",
        r"C:\Users\greeshmav\Tesseract-OCR\tesseract.exe"
    )
    
    # ── AWS / Region ──────────────────────────────────────────────────────────
    AWS_REGION:            str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID:     str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")

    # ── Model IDs ─────────────────────────────────────────────────────────────
    LLM_MODEL_ID:   str = os.getenv("LLM_MODEL_ID",   "meta.llama3-8b-instruct-v1:0")
    EMBED_MODEL_ID: str = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

    # ── Generation Parameters ─────────────────────────────────────────────────
    MAX_GEN_LEN_CHUNK:   int   = int(os.getenv("MAX_GEN_LEN_CHUNK",   350))
    MAX_GEN_LEN_SECTION: int   = int(os.getenv("MAX_GEN_LEN_SECTION", 400))
    MAX_GEN_LEN_EXEC:    int   = int(os.getenv("MAX_GEN_LEN_EXEC",    700))
    TEMPERATURE:         float = float(os.getenv("TEMPERATURE", 0.0))
    TOP_P:               float = float(os.getenv("TOP_P",       0.9))

    # ── Retry Settings ────────────────────────────────────────────────────────
    MAX_RETRIES_LLM:   int   = int(os.getenv("MAX_RETRIES_LLM",   3))
    MAX_RETRIES_EMBED: int   = int(os.getenv("MAX_RETRIES_EMBED", 2))
    BASE_DELAY:        float = float(os.getenv("BASE_DELAY", 0.8))

    # ── Clustering ────────────────────────────────────────────────────────────
    BASE_DISTANCE_RESEARCH:    float = float(os.getenv("BASE_DISTANCE_RESEARCH",    0.40))
    BASE_DISTANCE_ACADEMIC:    float = float(os.getenv("BASE_DISTANCE_ACADEMIC",    0.50))
    MIN_CHUNKS_FOR_CLUSTERING: int   = int(os.getenv("MIN_CHUNKS_FOR_CLUSTERING",   3))

    # ── Concurrency ───────────────────────────────────────────────────────────
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Neo4j (Knowledge Graph) ───────────────────────────────────────────────
    NEO4J_URI:      str = os.getenv("NEO4J_URI")
    NEO4J_USER:     str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")

    # ── Pinecone (Vector DB) ──────────────────────────────────────────────────
    PINECONE_API_KEY:   str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX:     str = os.getenv("PINECONE_INDEX", "docmind")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", 1024))
    PINECONE_METRIC:    str = os.getenv("PINECONE_METRIC", "cosine")

    # ── Retrieval / Search ────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))
    MAX_TOP_K:            int   = int(os.getenv("MAX_TOP_K", 5))

    # ── Vector Store Constants (NEW) ──────────────────────────────────────────
    UPSERT_BATCH_SIZE:          int = int(os.getenv("UPSERT_BATCH_SIZE", 100))
    METADATA_CHAR_LIMIT:        int = int(os.getenv("METADATA_CHAR_LIMIT", 3000))
    METADATA_CHAR_MIN_FALLBACK: int = int(os.getenv("METADATA_CHAR_MIN_FALLBACK", 500))
    MIN_CHUNK_TEXT_LEN:         int = int(os.getenv("MIN_CHUNK_TEXT_LEN", 80))
    MAX_SENTENCES_PER_CHUNK:    int = int(os.getenv("MAX_SENTENCES_PER_CHUNK", 8))
    MIN_SENTENCE_LEN:           int = int(os.getenv("MIN_SENTENCE_LEN", 30))
    MATH_SYMBOLS:               str = os.getenv(
        "MATH_SYMBOLS",
        "=∈∑→×·≤≥∗∝∂√∀∃∩∪⊂⊃±∞"
    )

    # ── Pipeline Controls ─────────────────────────────────────────────────────
    MAX_CHUNKS_PROCESS: int  = int(os.getenv("MAX_CHUNKS_PROCESS", 0))
    ENABLE_KG:          bool = os.getenv("ENABLE_KG",         "true").lower() == "true"
    ENABLE_EVALUATION:  bool = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"

    # ── Knowledge Graph Rules ─────────────────────────────────────────────────
    STRICT_RELATION_FILTERING: bool = (
        os.getenv("STRICT_RELATION_FILTERING", "true").lower() == "true"
    )

    # ── Timeouts ──────────────────────────────────────────────────────────────
    LLM_TIMEOUT:   int = int(os.getenv("LLM_TIMEOUT",   60))
    EMBED_TIMEOUT: int = int(os.getenv("EMBED_TIMEOUT", 30))

    # ── Environment ───────────────────────────────────────────────────────────
    ENV:   str  = os.getenv("ENV",   "dev")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    def validate(self) -> None:
        required = {
            "NEO4J_URI":              self.NEO4J_URI,
            "NEO4J_USER":             self.NEO4J_USER,
            "NEO4J_PASSWORD":         self.NEO4J_PASSWORD,
            "PINECONE_API_KEY":       self.PINECONE_API_KEY,
            "AWS_ACCESS_KEY_ID":      self.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY":  self.AWS_SECRET_ACCESS_KEY,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise EnvironmentError(
                "Missing required environment variables: %s" % ", ".join(missing)
            )


settings = Settings()