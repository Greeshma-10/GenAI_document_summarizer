import logging
import os
from v2.config import settings

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Main log file — all levels
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logging.getLogger().addHandler(console)

# Silence noisy third-party loggers
for noisy in ["boto3", "botocore", "urllib3", "neo4j", "httpx", "uvicorn.access"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("app")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger. Use __name__ as the name."""
    return logging.getLogger(f"app.{name}")