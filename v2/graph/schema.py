"""
Knowledge Graph Schema Definition

This file defines the allowed:
1. Entity types (node labels)
2. Relationship types (edge labels)
3. Triple structure used for graph construction
"""

from pydantic import BaseModel, validator
from typing import List


# ---------------------------------------------------
# Allowed Entity Types
# ---------------------------------------------------

ENTITY_TYPES: List[str] = [
    "PERSON",
    "ORGANIZATION",
    "METHOD",
    "DATASET",
    "TECHNOLOGY",
    "CONCEPT",
    "METRIC",
    "PAPER"
]


# ---------------------------------------------------
# Allowed Relationship Types
# ---------------------------------------------------

RELATION_TYPES: List[str] = [
    "DEVELOPED_BY",
    "PROPOSED_BY",
    "USES",
    "TRAINED_ON",
    "EVALUATED_ON",
    "WORKS_AT",
    "COLLABORATES_WITH",
    "RELATED_TO",
    "MENTIONS"
]


# ---------------------------------------------------
# Triple Data Model
# ---------------------------------------------------

class Triple(BaseModel):
    """
    Represents a Knowledge Graph triple.

    Example:
    ("BERT", "DEVELOPED_BY", "Google")
    """

    subject: str
    relation: str
    object: str

    @validator("relation")
    def validate_relation(cls, value):
        if value not in RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {value}")
        return value


# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------

def is_valid_entity(entity_type: str) -> bool:
    """
    Check if entity type is allowed.
    """
    return entity_type in ENTITY_TYPES


def is_valid_relation(relation: str) -> bool:
    """
    Check if relation type is allowed.
    """
    return relation in RELATION_TYPES