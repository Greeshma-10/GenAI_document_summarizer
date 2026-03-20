"""
Knowledge Graph Schema Definition
"""

from pydantic import BaseModel, validator
from typing import List


ENTITY_TYPES: List[str] = [
    "MODEL",
    "DATASET",
    "METRIC",
    "ORGANIZATION",
    "TASK",
    "CONCEPT",
    "Unknown"
]

RELATION_TYPES: List[str] = [
    "DEVELOPED_BY",
    "PROPOSED_BY",
    "USES",
    "USED_IN",          # ← ADDED: CONCEPT used in MODEL
    "TRAINED_ON",       # ← FIXED spelling (was TRAINING_ON)
    "EVALUATED_ON",
    "APPLIED_TO",
    "USED_FOR",
    "PART_OF",
    "SUPPORTS",
    "RELATED_TO",
    "MENTIONS"
]

CATEGORY_TO_SCHEMA_TYPE = {
    "models":        "MODEL",
    "datasets":      "DATASET",
    "metrics":       "METRIC",
    "organizations": "ORGANIZATION",
    "tasks":         "TASK",
    "key_concepts":  "CONCEPT"
}


class Triple(BaseModel):
    subject: str
    relation: str
    object: str
    subject_type: str = "Unknown"
    object_type: str = "Unknown"

    @validator("relation")
    def validate_relation(cls, value):
        if value not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {value}")
        return value

    @validator("subject_type", "object_type")
    def validate_entity_type(cls, value):
        if value not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {value}")
        return value


def is_valid_entity(entity_type: str) -> bool:
    return entity_type in ENTITY_TYPES


def is_valid_relation(relation: str) -> bool:
    return relation in RELATION_TYPES