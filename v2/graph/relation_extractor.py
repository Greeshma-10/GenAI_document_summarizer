"""
Relation Extractor

Extracts (subject, relation, object) triples from text using a Bedrock LLM,
then applies a validation and normalisation pipeline before returning
typed Triple objects ready for Neo4j insertion.
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

WEAK_RELATIONS = {"RELATED_TO", "MENTIONS"}

FLIP_RULES: Dict[tuple, str] = {
    ("METRIC",   "MODEL", "EVALUATED_ON"): "EVALUATED_ON",
    ("METRIC",   "MODEL", "USED_FOR"):     "EVALUATED_ON",
    ("DATASET",  "MODEL", "TRAINED_ON"):   "TRAINED_ON",
}

GARBAGE_OBJECTS = {"models", "evaluation", "systems", "the authors", "authors"}

_SUBJECT_MUST_BE_TYPED = {
    "DEVELOPED_BY", "PROPOSED_BY", "TRAINED_ON",
    "EVALUATED_ON", "APPLIED_TO", "PART_OF",
}

# LLM sometimes returns entity/category names as relation types — block them
_ENTITY_TYPE_NAMES = {
    "CONCEPT", "MODEL", "DATASET", "METRIC",
    "ORGANIZATION", "TASK", "ENTITY", "TYPE",
}


# ─────────────────────────────────────────────────────────────────────────────
class RelationExtractor:

    def __init__(self):
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
        body = {
            "prompt":      prompt,
            "max_gen_len": 1024,
            "temperature": 0.0,
            "top_p":       0.9,
        }
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result.get("generation", result.get("outputs", [{}])[0].get("text", ""))
        except Exception:
            logger.exception("Bedrock invocation failed")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    def _clean_json(self, text: str) -> Optional[str]:
        text = re.sub(r"```(?:json)?", "", text).strip().replace("```", "").strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            return match.group()
        objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
        if objects:
            return "[" + ",".join(objects) + "]"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _build_entity_type_map(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        entity_map: Dict[str, str] = {}
        for category, values in entities.items():
            entity_type = CATEGORY_TO_SCHEMA_TYPE.get(category, "Unknown")
            for v in values:
                if v:
                    entity_map[v] = entity_type
        return entity_map

    # ─────────────────────────────────────────────────────────────────────────
    def _normalize(self, text: str) -> str:
        return text.lower().strip().replace("-", " ")

    # ─────────────────────────────────────────────────────────────────────────
    def _match_entity(self, text: str, normalized_to_canonical: Dict[str, str]) -> Optional[str]:
        text_norm = self._normalize(text)
        if text_norm in normalized_to_canonical:
            return normalized_to_canonical[text_norm]
        for key in normalized_to_canonical:
            if text_norm in key or key in text_norm:
                return normalized_to_canonical[key]
        text_words = set(text_norm.split())
        for key in normalized_to_canonical:
            if len(text_words & set(key.split())) >= 1:
                return normalized_to_canonical[key]
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _upgrade_relation(self, relation: str, subject_type: str, object_type: str) -> str:
        if relation in WEAK_RELATIONS:
            preferred = PREFERRED_RELATION.get((subject_type, object_type))
            if preferred:
                logger.debug("Upgraded relation %s -> %s (%s -> %s)",
                             relation, preferred, subject_type, object_type)
                return preferred
        return relation

    # ─────────────────────────────────────────────────────────────────────────
    def _maybe_flip(self, subject, relation, obj, subject_type, object_type):
        flip_relation = FLIP_RULES.get((subject_type, object_type, relation))
        if flip_relation:
            logger.debug("Flipped triple: (%s:%s)-[%s]->(%s:%s)",
                         subject, subject_type, relation, obj, object_type)
            return obj, flip_relation, subject, object_type, subject_type
        return subject, relation, obj, subject_type, object_type

    # ─────────────────────────────────────────────────────────────────────────
    def extract_relations(self, text: str, entities: Dict[str, List[str]]) -> List[Triple]:

        if not entities:
            return []

        entity_type_map = self._build_entity_type_map(entities)
        if not entity_type_map:
            logger.warning("No entities provided for relation extraction")
            return []

        normalized_to_canonical = {self._normalize(k): k for k in entity_type_map}

        entities_with_types = "\n".join(
            f"  {name} - {etype}" for name, etype in entity_type_map.items()
        )

        prompt        = build_relation_extraction_prompt(text, entities_with_types)
        response_text = self._invoke_bedrock(prompt)

        if not response_text:
            logger.warning("Empty response from Bedrock in relation extraction")
            return []

        logger.debug("Bedrock response (first 300 chars): %s", response_text[:300])

        json_text = self._clean_json(response_text)
        if not json_text:
            logger.warning("No JSON found in relation extraction response")
            return []

        try:
            triples_json = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed | snippet=%s", json_text[:200])
            return []

        if isinstance(triples_json, dict):
            triples_json = [triples_json]
        if not isinstance(triples_json, list):
            logger.warning("Unexpected JSON structure in relation extraction response")
            return []

        seen    = set()
        triples = []

        for item in triples_json:
            try:
                subject_raw = item.get("subject", "").strip()
                object_raw  = item.get("object",  "").strip()
                relation    = item.get("relation", "").upper().strip()

                if not subject_raw or not object_raw or not relation:
                    continue

                subject_canonical = self._match_entity(subject_raw, normalized_to_canonical)
                obj_canonical     = self._match_entity(object_raw,  normalized_to_canonical)

                logger.debug("Match subject: '%s' -> '%s'", subject_raw, subject_canonical)
                logger.debug("Match object:  '%s' -> '%s'", object_raw,  obj_canonical)

                if not subject_canonical:
                    logger.debug("Skipping: subject not matched - %s", subject_raw)
                    continue

                if not obj_canonical:
                    if len(object_raw.split()) <= 3:
                        obj_canonical = object_raw
                        logger.debug("Raw object fallback: '%s'", obj_canonical)
                    else:
                        logger.debug("Skipping: object not matched and too long - %s", object_raw)
                        continue

                if obj_canonical.lower() in GARBAGE_OBJECTS:
                    logger.debug("Skipping: garbage object '%s'", obj_canonical)
                    continue

                if len(obj_canonical.split()) > 10:
                    logger.debug("Skipping: object too long - %s", obj_canonical)
                    continue

                subject_type = entity_type_map.get(subject_canonical, "Unknown")
                object_type  = entity_type_map.get(obj_canonical,     "Unknown")

                # Step 1: upgrade weak relations
                relation = self._upgrade_relation(relation, subject_type, object_type)

                # Step 2: flip backwards triples
                subject_canonical, relation, obj_canonical, subject_type, object_type = \
                    self._maybe_flip(subject_canonical, relation, obj_canonical,
                                     subject_type, object_type)

                # Step 3: drop relations that are entity/category type names
                # The LLM occasionally returns "CONCEPT", "MODEL" etc. as relation types
                if relation in _ENTITY_TYPE_NAMES:
                    logger.debug("Skipping: relation is an entity type name - %s", relation)
                    continue

                # Step 4: validate relation — predefined OR UPPERCASE_WITH_UNDERSCORES
                is_valid = (relation in RELATION_TYPES or
                            bool(re.match(r'^[A-Z][A-Z_]+$', relation)))
                if not is_valid:
                    logger.debug("Skipping: invalid relation format - %s", relation)
                    continue

                if relation not in RELATION_TYPES:
                    logger.info("New relation type discovered: %s", relation)

                # Step 5: drop MODEL RELATED_TO TASK — too generic
                if relation == "RELATED_TO" and subject_type == "MODEL" and object_type == "TASK":
                    logger.debug("Skipping: MODEL RELATED_TO TASK is too generic")
                    continue

                # Step 6: drop self-loops
                if subject_canonical.lower() == obj_canonical.lower():
                    logger.debug("Skipping: self-loop - %s", subject_canonical)
                    continue

                # Step 7: DEVELOPED_BY / PROPOSED_BY object must be ORGANIZATION
                if relation in {"DEVELOPED_BY", "PROPOSED_BY"}:
                    obj_type_check = entity_type_map.get(obj_canonical, "Unknown")
                    if obj_type_check not in {"ORGANIZATION", "Unknown"}:
                        logger.debug("Skipping: %s object '%s' is %s not ORGANIZATION",
                                     relation, obj_canonical, obj_type_check)
                        continue

                # Step 8: drop publication names as subjects
                if re.search(r'\[|\bJournal\b|\bInternational\b|\bConference\b',
                             subject_canonical, re.IGNORECASE):
                    logger.debug("Skipping: publication-name subject - %s", subject_canonical)
                    continue

                # Step 9: subject must be typed for structural relations
                if relation in _SUBJECT_MUST_BE_TYPED:
                    type_map_lower = {k.lower(): v for k, v in entity_type_map.items()}
                    if subject_canonical.lower() not in type_map_lower:
                        logger.debug("Skipping: untyped subject '%s' for relation %s",
                                     subject_canonical, relation)
                        continue

                # Step 10: deduplicate
                key = (subject_canonical, relation, obj_canonical)
                if key in seen:
                    logger.debug("Skipping: duplicate triple")
                    continue
                seen.add(key)

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
                logger.exception("Error processing triple item: %s", item)
                continue

        logger.info("Relation extraction complete: %d triples", len(triples))
        return triples