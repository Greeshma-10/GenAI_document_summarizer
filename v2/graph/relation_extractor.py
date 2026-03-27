"""
Relation Extractor
"""

import json
import re
import boto3
from typing import Dict, List, Optional

from v2.graph.schema import Triple, RELATION_TYPES, CATEGORY_TO_SCHEMA_TYPE
from v2.config import settings
from v2.prompts.relation import build_relation_extraction_prompt
from v2.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Relation upgrade / flip / filter tables
# ─────────────────────────────────────────────────────────────────────────────

# Preferred stronger relations based on subject-object type pairs
PREFERRED_RELATION: Dict[tuple, str] = {
    ("MODEL",        "ORGANIZATION"): "DEVELOPED_BY",
    ("MODEL",        "DATASET"):      "TRAINED_ON",
    ("MODEL",        "METRIC"):       "EVALUATED_ON",
    ("MODEL",        "TASK"):         "APPLIED_TO",
    ("MODEL",        "CONCEPT"):      "USES",
    ("CONCEPT",      "MODEL"):        "USED_IN",
    ("CONCEPT",      "CONCEPT"):      "RELATED_TO",
    ("METRIC",       "TASK"):         "USED_FOR",
    ("METRIC",       "MODEL"):        "USED_FOR",
    ("DATASET",      "TASK"):         "USED_FOR",
    ("ORGANIZATION", "MODEL"):        "PROPOSED_BY",
}

# Weak relations that can be upgraded to stronger ones
WEAK_RELATIONS = {"RELATED_TO", "MENTIONS"}

# Rules to flip subject-object direction for certain relations
FLIP_RULES: Dict[tuple, str] = {
    ("METRIC",   "MODEL", "EVALUATED_ON"): "EVALUATED_ON",
    ("METRIC",   "MODEL", "USED_FOR"):     "EVALUATED_ON",
    ("DATASET",  "MODEL", "TRAINED_ON"):   "TRAINED_ON",
}

# Common low-quality or meaningless object values to ignore
GARBAGE_OBJECTS = {"models", "evaluation", "systems", "the authors", "authors"}

# Relations that require subject to have a known type
_SUBJECT_MUST_BE_TYPED = {
    "DEVELOPED_BY", "PROPOSED_BY", "TRAINED_ON",
    "EVALUATED_ON", "APPLIED_TO", "PART_OF",
}

# Prevent entity type names from being used as relation names
_ENTITY_TYPE_NAMES = {
    "CONCEPT", "MODEL", "DATASET", "METRIC",
    "ORGANIZATION", "TASK", "ENTITY", "TYPE",
}


# ─────────────────────────────────────────────────────────────────────────────
class RelationExtractor:

    def __init__(self):
        # Initialize AWS Bedrock client for LLM invocation
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        self.model_id = settings.LLM_MODEL_ID
        logger.info("RelationExtractor initialized | model=%s", self.model_id)

    # ─────────────────────────────────────────────────────────────────────────
    def _invoke_bedrock(self, prompt: str) -> str:
        # Prepare request payload for LLM
        body = {
            "prompt":      prompt,
            "max_gen_len": 1024,
            "temperature": 0.0,
            "top_p":       0.9,
        }
        try:
            # Call Bedrock model
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            # Extract generated text from response
            result = json.loads(response["body"].read())
            return result.get("generation", result.get("outputs", [{}])[0].get("text", ""))
        except Exception:
            # Log failure and return empty string
            logger.exception("Bedrock invocation failed")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    def _clean_json(self, text: str) -> Optional[str]:
        # Remove markdown formatting and extract JSON-like content
        text = re.sub(r"```(?:json)?", "", text).strip().replace("```", "").strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return match.group()
        # Fallback: extract individual objects and wrap as list
        objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
        if objects:
            return "[" + ",".join(objects) + "]"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _build_entity_type_map(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        # Map each entity name to its schema type
        entity_map: Dict[str, str] = {}
        for category, values in entities.items():
            entity_type = CATEGORY_TO_SCHEMA_TYPE.get(category, "Unknown")
            for v in values:
                if v:
                    entity_map[v] = entity_type
        return entity_map

    # ─────────────────────────────────────────────────────────────────────────
    def _normalize(self, text: str) -> str:
        # Normalize text for matching (lowercase, remove hyphens)
        return text.lower().strip().replace("-", " ")

    # ─────────────────────────────────────────────────────────────────────────
    def _match_entity(self, text: str, normalized_to_canonical: Dict[str, str]) -> Optional[str]:
        # Normalize input text
        text_norm = self._normalize(text)

        # Exact match
        if text_norm in normalized_to_canonical:
            return normalized_to_canonical[text_norm]

        # Partial substring match
        for key in normalized_to_canonical:
            if text_norm in key or key in text_norm:
                return normalized_to_canonical[key]

        # Word overlap match
        text_words = set(text_norm.split())
        for key in normalized_to_canonical:
            if len(text_words & set(key.split())) >= 1:
                return normalized_to_canonical[key]

        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _upgrade_relation(self, relation: str, subject_type: str, object_type: str) -> str:
        # Upgrade weak relations to stronger predefined ones
        if relation in WEAK_RELATIONS:
            preferred = PREFERRED_RELATION.get((subject_type, object_type))
            if preferred:
                logger.debug("Upgraded relation %s -> %s (%s -> %s)",
                             relation, preferred, subject_type, object_type)
                return preferred
        return relation

    # ─────────────────────────────────────────────────────────────────────────
    def _maybe_flip(self, subject, relation, obj, subject_type, object_type):
        # Flip subject-object if rule exists
        flip_relation = FLIP_RULES.get((subject_type, object_type, relation))
        if flip_relation:
            logger.debug("Flipped triple: (%s:%s)-[%s]->(%s:%s)",
                         subject, subject_type, relation, obj, object_type)
            return obj, flip_relation, subject, object_type, subject_type
        return subject, relation, obj, subject_type, object_type

    # ─────────────────────────────────────────────────────────────────────────
    def extract_relations(self, text: str, entities: Dict[str, List[str]]) -> List[Triple]:

        # Return empty if no entities provided
        if not entities:
            return []

        # Build entity-to-type mapping
        entity_type_map = self._build_entity_type_map(entities)
        if not entity_type_map:
            logger.warning("No entities provided for relation extraction")
            return []

        # Normalize entity names for matching
        normalized_to_canonical = {self._normalize(k): k for k in entity_type_map}

        # Format entities for prompt
        entities_with_types = "\n".join(
            f"  {name} - {etype}" for name, etype in entity_type_map.items()
        )

        # Build and send prompt to LLM
        prompt        = build_relation_extraction_prompt(text, entities_with_types)
        response_text = self._invoke_bedrock(prompt)

        # Handle empty LLM response
        if not response_text:
            logger.warning("Empty response from Bedrock in relation extraction")
            return []

        logger.debug("Bedrock response (first 300 chars): %s", response_text[:300])

        # Extract JSON from response
        json_text = self._clean_json(response_text)
        if not json_text:
            logger.warning("No JSON found in relation extraction response")
            return []

        # Parse JSON safely
        try:
            triples_json = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed | snippet=%s", json_text[:200])
            return []

        # Normalize structure into list
        if isinstance(triples_json, dict):
            triples_json = [triples_json]
        if not isinstance(triples_json, list):
            logger.warning("Unexpected JSON structure in relation extraction response")
            return []

        seen    = set()   # Track duplicates
        triples = []      # Store final triples

        for item in triples_json:
            try:
                # Extract fields from JSON
                subject_raw = item.get("subject", "").strip()
                object_raw  = item.get("object",  "").strip()
                relation    = item.get("relation", "").upper().strip()

                # Skip incomplete entries
                if not subject_raw or not object_raw or not relation:
                    continue

                # Match entities to canonical names
                subject_canonical = self._match_entity(subject_raw, normalized_to_canonical)
                obj_canonical     = self._match_entity(object_raw,  normalized_to_canonical)

                logger.debug("Match subject: '%s' -> '%s'", subject_raw, subject_canonical)
                logger.debug("Match object:  '%s' -> '%s'", object_raw,  obj_canonical)

                # Skip if subject not matched
                if not subject_canonical:
                    logger.debug("Skipping: subject not matched - %s", subject_raw)
                    continue

                # Handle unmatched object
                if not obj_canonical:
                    if len(object_raw.split()) <= 3:
                        obj_canonical = object_raw  # fallback
                        logger.debug("Raw object fallback: '%s'", obj_canonical)
                    else:
                        logger.debug("Skipping: object not matched and too long - %s", object_raw)
                        continue

                # Filter garbage objects
                if obj_canonical.lower() in GARBAGE_OBJECTS:
                    logger.debug("Skipping: garbage object '%s'", obj_canonical)
                    continue

                # Skip overly long objects
                if len(obj_canonical.split()) > 10:
                    logger.debug("Skipping: object too long - %s", obj_canonical)
                    continue

                # Get entity types
                subject_type = entity_type_map.get(subject_canonical, "Unknown")
                object_type  = entity_type_map.get(obj_canonical,     "Unknown")

                # Step 1: upgrade weak relations
                relation = self._upgrade_relation(relation, subject_type, object_type)

                # Step 2: flip relations if needed
                subject_canonical, relation, obj_canonical, subject_type, object_type = \
                    self._maybe_flip(subject_canonical, relation, obj_canonical,
                                     subject_type, object_type)

                # Step 3: remove invalid relation types
                if relation in _ENTITY_TYPE_NAMES:
                    logger.debug("Skipping: relation is an entity type name - %s", relation)
                    continue

                # Step 4: validate relation format
                is_valid = (relation in RELATION_TYPES or
                            bool(re.match(r'^[A-Z][A-Z_]+$', relation)))
                if not is_valid:
                    logger.debug("Skipping: invalid relation format - %s", relation)
                    continue

                # Log newly discovered relations
                if relation not in RELATION_TYPES:
                    logger.info("New relation type discovered: %s", relation)

                # Step 5: filter generic MODEL-TASK relations
                if relation == "RELATED_TO" and subject_type == "MODEL" and object_type == "TASK":
                    logger.debug("Skipping: MODEL RELATED_TO TASK is too generic")
                    continue

                # Step 6: remove self-loops
                if subject_canonical.lower() == obj_canonical.lower():
                    logger.debug("Skipping: self-loop - %s", subject_canonical)
                    continue

                # Step 7: enforce ORGANIZATION for certain relations
                if relation in {"DEVELOPED_BY", "PROPOSED_BY"}:
                    obj_type_check = entity_type_map.get(obj_canonical, "Unknown")
                    if obj_type_check not in {"ORGANIZATION", "Unknown"}:
                        logger.debug("Skipping: %s object '%s' is %s not ORGANIZATION",
                                     relation, obj_canonical, obj_type_check)
                        continue

                # Step 8: filter publication-like subjects
                if re.search(r'\[|\bJournal\b|\bInternational\b|\bConference\b',
                             subject_canonical, re.IGNORECASE):
                    logger.debug("Skipping: publication-name subject - %s", subject_canonical)
                    continue

                # Step 9: ensure typed subjects for critical relations
                if relation in _SUBJECT_MUST_BE_TYPED:
                    type_map_lower = {k.lower(): v for k, v in entity_type_map.items()}
                    if subject_canonical.lower() not in type_map_lower:
                        logger.debug("Skipping: untyped subject '%s' for relation %s",
                                     subject_canonical, relation)
                        continue

                # Step 10: remove duplicates
                key = (subject_canonical, relation, obj_canonical)
                if key in seen:
                    logger.debug("Skipping: duplicate triple")
                    continue
                seen.add(key)

                # Create Triple object
                triple = Triple(
                    subject=subject_canonical,
                    relation=relation,
                    object=obj_canonical,
                    subject_type=subject_type,
                    object_type=object_type,
                )
                triples.append(triple)

                logger.debug("Added: (%s:%s)-[%s]->(%s:%s)",
                             subject_canonical, subject_type, relation,
                             obj_canonical, object_type)

            except Exception:
                # Catch and log errors for each item
                logger.exception("Error processing triple item: %s", item)
                continue

        # Final log with total triples extracted
        logger.info("Relation extraction complete: %d triples", len(triples))
        return triples