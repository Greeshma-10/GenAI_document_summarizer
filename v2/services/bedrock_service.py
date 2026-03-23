"""
Bedrock Service

Provides two core functions for interacting with AWS Bedrock:
  1. invoke_llm()    — prompt a text-generation model and return parsed JSON
  2. get_embedding() — embed a string and return the embedding vector
"""

import json
import time
from typing import Dict, List, Optional

import boto3

from v1.config import settings
from v2.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

client = boto3.client(
    service_name="bedrock-runtime",
    region_name=settings.AWS_REGION,
    aws_access_key_id=getattr(settings, "AWS_ACCESS_KEY_ID", None),
    aws_secret_access_key=getattr(settings, "AWS_SECRET_ACCESS_KEY", None),
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_parse_json(text: str) -> Dict:
    """
    Extract and parse the first valid JSON object found in `text`.
    Raises ValueError if no valid JSON can be recovered.
    """
    if not text:
        raise ValueError("Empty model response")

    text  = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    text = text[start:]
    for i in range(len(text), 0, -1):
        try:
            return json.loads(text[:i])
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not extract valid JSON from model response")


# ─────────────────────────────────────────────────────────────────────────────
# LLM invocation
# ─────────────────────────────────────────────────────────────────────────────

def invoke_llm(
    prompt: str,
    max_gen_len: int,
    stop_tokens: Optional[List[str]] = None,
) -> Dict:
    """
    Invoke the configured Bedrock LLM and return the parsed JSON response.

    Retries up to settings.MAX_RETRIES_LLM times on failure.
    Raises RuntimeError if all attempts are exhausted.
    """
    if not prompt:
        raise ValueError("Prompt must not be empty")

    body = {
        "prompt":      prompt,
        "max_gen_len": max_gen_len,
        "temperature": settings.TEMPERATURE,
        "top_p":       settings.TOP_P,
        "stop":        stop_tokens or ["END"],
    }

    start = time.time()

    for attempt in range(settings.MAX_RETRIES_LLM + 1):
        try:
            response = client.invoke_model(
                modelId=settings.LLM_MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body  = json.loads(response["body"].read())
            generated_text = response_body.get("generation", "").strip()

            logger.debug("LLM raw response (first 300 chars): %s", generated_text[:300])

            parsed  = _safe_parse_json(generated_text)
            latency = round(time.time() - start, 2)

            logger.info("LLM invocation successful | latency=%.2fs attempt=%d model=%s",
                        latency, attempt + 1, settings.LLM_MODEL_ID)

            return parsed

        except Exception:
            logger.warning("LLM attempt %d/%d failed",
                           attempt + 1, settings.MAX_RETRIES_LLM + 1)
            time.sleep(settings.BASE_DELAY)

    raise RuntimeError("LLM invocation failed after %d attempts" % (settings.MAX_RETRIES_LLM + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Embedding invocation
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    """
    Embed `text` using the configured Bedrock embedding model.

    Retries up to settings.MAX_RETRIES_EMBED times on failure.
    Raises RuntimeError if all attempts are exhausted.
    """
    body = {"inputText": text}

    for attempt in range(settings.MAX_RETRIES_EMBED + 1):
        try:
            response = client.invoke_model(
                modelId=settings.EMBED_MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(response["body"].read())
            logger.debug("Embedding successful | model=%s attempt=%d",
                         settings.EMBED_MODEL_ID, attempt + 1)
            return result["embedding"]

        except Exception:
            logger.warning("Embedding attempt %d/%d failed",
                           attempt + 1, settings.MAX_RETRIES_EMBED + 1)
            time.sleep(settings.BASE_DELAY)

    raise RuntimeError("Embedding failed after %d attempts" % (settings.MAX_RETRIES_EMBED + 1))