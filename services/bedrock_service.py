import boto3 # type: ignore
import json
import time
from typing import Dict, List
from config import settings
from logger import logger
import time

# CLIENT INITIALIZATION

client = boto3.client(
    service_name="bedrock-runtime",
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID if hasattr(settings, "AWS_ACCESS_KEY_ID") else None,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if hasattr(settings, "AWS_SECRET_ACCESS_KEY") else None
)

# SAFE JSON PARSER (Unified)

def safe_parse_json(text: str) -> Dict:
    if not text:
        raise ValueError("Empty model response")

    text = text.strip()
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON found in model response")

    text = text[start:]

    for i in range(len(text), 0, -1):
        try:
            return json.loads(text[:i])
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not extract valid JSON")


# LLM INVOCATION

def invoke_llm(
    prompt: str,
    max_gen_len: int,
    stop_tokens: List[str] = None
) -> Dict:
    start = time.time()
    if not prompt:
        raise ValueError("Prompt is empty")

    body = {
        "prompt": prompt,
        "max_gen_len": max_gen_len,
        "temperature": settings.TEMPERATURE,
        "top_p": settings.TOP_P,
        "stop": stop_tokens or ["```", "END"]
    }

    for attempt in range(settings.MAX_RETRIES_LLM + 1):

        try:
            response = client.invoke_model(
                modelId=settings.LLM_MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            generated_text = response_body.get("generation", "").strip()

            parsed = safe_parse_json(generated_text)

            return parsed

        except Exception as e:
            logger.warning(
                f"LLM attempt {attempt+1} failed: {str(e)}"
            )
            time.sleep(settings.BASE_DELAY)

    raise RuntimeError("LLM failed after retries")
    end = time.time()
    logger.info(f"LLM call latency: {round(end-start,2)} sec")

# EMBEDDING INVOCATION

def get_embedding(text: str) -> List[float]:

    body = {"inputText": text}

    for attempt in range(settings.MAX_RETRIES_EMBED + 1):

        try:
            response = client.invoke_model(
                modelId=settings.EMBED_MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )

            result = json.loads(response["body"].read())
            return result["embedding"]

        except Exception as e:
            logger.warning(
                f"Embedding attempt {attempt+1} failed: {str(e)}"
            )
            time.sleep(settings.BASE_DELAY)

    raise RuntimeError("Embedding failed after retries")