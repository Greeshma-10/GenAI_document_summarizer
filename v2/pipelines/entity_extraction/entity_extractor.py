"""
Entity Extractor
"""

from v2.services.bedrock_service import invoke_llm
from v2.prompts.entity import build_entity_extraction_prompt
from v2.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_KEYS = ["models", "datasets", "metrics", "organizations", "tasks", "key_concepts"]

EMPTY_RESULT = {k: [] for k in EXPECTED_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
def extract_entities(text: str) -> dict:
    """
    Extract structured entities from the provided text.

    Returns a dict with keys: models, datasets, metrics,
    organizations, tasks, key_concepts — each a list of strings.
    """
    prompt = build_entity_extraction_prompt(text)

    try:
        entities = invoke_llm(prompt, max_gen_len=600)
        logger.debug("LLM raw response: %s", str(entities)[:300])

        # Ensure all expected keys exist and values are flat strings
        for key in EXPECTED_KEYS:
            if key not in entities:
                entities[key] = []
            entities[key] = [
                str(v).strip() for v in entities[key]
                if v and str(v).strip()
            ]

        orgs = entities.get("organizations", [])
        if orgs:
            logger.info("Organizations extracted: %s", orgs)
        else:
            logger.warning("No organizations extracted from chunk")

        logger.info(
            "Entity extraction complete | %s",
            " | ".join(f"{k}=%d" % len(entities[k]) for k in EXPECTED_KEYS),
        )
        return entities

    except Exception:
        logger.exception("Entity extraction failed")
        return EMPTY_RESULT