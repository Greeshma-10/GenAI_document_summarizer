"""
Entity Utilities

flatten_entities: converts the categorized entity dict from the LLM into
the format relation_extractor expects.

LLM produces:
  {
    "models":        ["Transformer", "ByteNet"],
    "organizations": ["Google Brain", "University of Toronto"],
    "metrics":       ["BLEU", "F1"],
    ...
  }

relation_extractor expects exactly this dict — DO NOT flatten to a plain list.
The extractor uses the category keys to look up schema types via CATEGORY_TO_SCHEMA_TYPE.
"""

from typing import Dict, List, Union


def flatten_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Pass the categorized entity dict through unchanged.
    Filters out empty lists and blank strings.

    Returns the same dict structure the relation_extractor needs:
      { "models": [...], "organizations": [...], ... }
    """
    if not entities:
        return {}

    cleaned: Dict[str, List[str]] = {}
    for category, values in entities.items():
        if not isinstance(values, list):
            continue
        filtered = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        if filtered:
            cleaned[category] = filtered

    return cleaned


def flatten_entities_to_list(entities: Dict[str, List[str]]) -> List[str]:
    """
    Flatten to a plain list of entity name strings.
    Use this ONLY for display/logging — NOT for relation extraction.
    """
    result = []
    for values in entities.values():
        if isinstance(values, list):
            result.extend(v.strip() for v in values if isinstance(v, str) and v.strip())
    return result