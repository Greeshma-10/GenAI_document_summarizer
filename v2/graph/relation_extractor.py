"""
Relation Extraction Module — Bedrock Version

Changes from Ollama version:
✔ Uses AWS Bedrock (meta.llama3-8b-instruct-v1:0) instead of local Ollama
✔ Faster and more accurate than local LLaMA
✔ Uses same bedrock_service pattern as entity_extractor
✔ All existing fixes preserved (flip rules, upgrade, dedup, filters)
"""

import json
import re
import boto3
from typing import List, Dict, Optional

from v2.graph.schema import Triple, RELATION_TYPES, CATEGORY_TO_SCHEMA_TYPE
# RELATION_TYPES used as preferred list only — LLM may define new types
from v2.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Preferred relation for (subject_type, object_type) pairs
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
    ("METRIC", "MODEL", "EVALUATED_ON"): "EVALUATED_ON",
    ("METRIC", "MODEL", "USED_FOR"):     "EVALUATED_ON",
    ("DATASET","MODEL", "TRAINED_ON"):   "TRAINED_ON",
}

PROMPT_TEMPLATE = """
You are a precise knowledge graph extraction system for scientific papers.

Extract factual relationships between the listed entities from the text.

Text:
{text}

Entities (Name — Type):
{entities}

PREFERRED relations (use these when they fit):
- DEVELOPED_BY   : model/tool created by an organization or person
- PROPOSED_BY    : model/concept introduced/proposed by an organization
- TRAINED_ON     : model trained using a dataset
- EVALUATED_ON   : model tested/benchmarked on a metric or dataset
- APPLIED_TO     : model/method used for a task
- USES           : model/system uses a concept/mechanism
- USED_IN        : concept/technique used inside a model/system
- USED_FOR       : metric/dataset used for a specific purpose
- PART_OF        : component that belongs to a larger system
- OUTPERFORMS    : model achieves better results than another model
- COMPARED_TO    : model explicitly compared against another model
- RELATED_TO     : general relationship when nothing specific fits

You MAY define a NEW relation type if none of the above fits and the relationship
is clearly stated in the text. Use UPPERCASE_WITH_UNDERSCORES format.
Examples of valid new relations: INTRODUCES, REPLACES, EXTENDS, IMPROVES_UPON

Special rules for tables and results:
- "Transformer (big) 28.4 BLEU EN-DE" → subject: Transformer (big), relation: EVALUATED_ON, object: BLEU
- Model comparison rows → extract EVALUATED_ON for each model/metric pair
- "X outperforms Y" → subject: X, relation: OUTPERFORMS, object: Y
- Training data mentioned → extract TRAINED_ON

Rules:
- Only extract relationships explicitly stated or strongly implied in the text
- Do NOT invent relationships that are not in the text
- Be specific — prefer EVALUATED_ON over RELATED_TO for metrics
- Skip generic or obvious triples
- Return ONLY a valid JSON array, no explanation, no markdown

Output format:
[
  {{"subject": "Entity1", "relation": "RELATION", "object": "Entity2"}}
]
"""


class RelationExtractor:

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        self.model_id = settings.LLM_MODEL_ID
        print(f"✅ RelationExtractor using Bedrock: {self.model_id}")

    # ─────────────────────────────────────────────────────────────────────────
    def _invoke_bedrock(self, prompt: str) -> str:
        """Call Bedrock LLaMA3 and return the text response."""
        body = {
            "prompt": prompt,
            "max_gen_len": 1024,
            "temperature": 0.0,
            "top_p": 0.9,
        }
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            # LLaMA3 on Bedrock returns generation field
            return result.get("generation", result.get("outputs", [{}])[0].get("text", ""))
        except Exception as e:
            print(f"⚠️ Bedrock invocation failed: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    def _clean_json(self, text: str) -> Optional[str]:
        text = re.sub(r"```(?:json)?", "", text).strip()
        text = text.replace("```", "").strip()

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
                print(f"   ⬆️  Upgraded: {relation} → {preferred} ({subject_type}→{object_type})")
                return preferred
        return relation

    # ─────────────────────────────────────────────────────────────────────────
    def _maybe_flip(self, subject, relation, obj, subject_type, object_type):
        flip_relation = FLIP_RULES.get((subject_type, object_type, relation))
        if flip_relation:
            print(f"   🔄 Flipped: ({subject}:{subject_type}) -[{relation}]-> "
                  f"({obj}:{object_type})  →  ({obj}:{object_type}) "
                  f"-[{flip_relation}]-> ({subject}:{subject_type})")
            return obj, flip_relation, subject, object_type, subject_type
        return subject, relation, obj, subject_type, object_type

    # ─────────────────────────────────────────────────────────────────────────
    def extract_relations(self, text: str, entities: Dict[str, List[str]]) -> List[Triple]:

        if not entities:
            return []

        entity_type_map = self._build_entity_type_map(entities)
        if not entity_type_map:
            print("⚠️ No entities to extract relations from.")
            return []

        normalized_to_canonical = {
            self._normalize(k): k for k in entity_type_map
        }

        entities_with_types = "\n".join(
            f"  {name} — {etype}" for name, etype in entity_type_map.items()
        )

        prompt = PROMPT_TEMPLATE.format(
            text=text,
            entities=entities_with_types
        )

        response_text = self._invoke_bedrock(prompt)

        if not response_text:
            print("⚠️ Empty response from Bedrock")
            return []

        print("\n🔗 RELATION BEDROCK RESPONSE:\n", response_text[:500])

        json_text = self._clean_json(response_text)
        if not json_text:
            print("⚠️ No JSON detected in response")
            return []

        try:
            triples_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed: {e}")
            print(f"   Raw snippet: {json_text[:300]}")
            return []

        if isinstance(triples_json, dict):
            triples_json = [triples_json]
        if not isinstance(triples_json, list):
            print("⚠️ Unexpected JSON structure")
            return []

        GARBAGE_OBJECTS = {"models", "evaluation", "systems", "the authors", "authors"}
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

                print(f"🔍 MATCH subject: '{subject_raw}' → '{subject_canonical}'")
                print(f"🔍 MATCH object:  '{object_raw}'  → '{obj_canonical}'")

                if not subject_canonical:
                    print(f"   ⏭ Skipping: subject not matched")
                    continue

                if not obj_canonical:
                    if len(object_raw.split()) <= 3:
                        obj_canonical = object_raw
                        print(f"   ℹ️ Raw object fallback: '{obj_canonical}'")
                    else:
                        print(f"   ⏭ Skipping: object not matched and too long")
                        continue

                if obj_canonical.lower() in GARBAGE_OBJECTS:
                    print(f"   ⏭ Skipping: garbage object '{obj_canonical}'")
                    continue

                if len(obj_canonical.split()) > 10:
                    print(f"   ⏭ Skipping: object too long")
                    continue

                subject_type = entity_type_map.get(subject_canonical, "Unknown")
                object_type  = entity_type_map.get(obj_canonical,     "Unknown")

                # Step 1: upgrade weak relations
                relation = self._upgrade_relation(relation, subject_type, object_type)

                # Step 2: flip backwards triples
                subject_canonical, relation, obj_canonical, subject_type, object_type = \
                    self._maybe_flip(subject_canonical, relation, obj_canonical,
                                     subject_type, object_type)

                # Step 3: validate relation
                if relation not in RELATION_TYPES:
                    print(f"   ⏭ Skipping: invalid relation '{relation}'")
                    continue

                # Step 4: drop MODEL RELATED_TO TASK
                if relation == "RELATED_TO" and subject_type == "MODEL" and object_type == "TASK":
                    print(f"   ⏭ Skipping: MODEL RELATED_TO TASK is too generic")
                    continue

                # Step 5: drop self-loops
                if subject_canonical.lower() == obj_canonical.lower():
                    print(f"   ⏭ Skipping: self-loop")
                    continue

                # Step 6: DEVELOPED_BY / PROPOSED_BY object must be ORGANIZATION
                if relation in {"DEVELOPED_BY", "PROPOSED_BY"}:
                    obj_type_check = entity_type_map.get(obj_canonical, "Unknown")
                    if obj_type_check not in {"ORGANIZATION", "Unknown"}:
                        print(f"   ⏭ Skipping: {relation} object '{obj_canonical}' "
                              f"is {obj_type_check}, not ORGANIZATION")
                        continue

                # Step 7: drop publication names as subjects
                if re.search(r'\[|\bJournal\b|\bInternational\b|\bConference\b',
                             subject_canonical, re.IGNORECASE):
                    print(f"   ⏭ Skipping: subject looks like a publication")
                    continue

                # Step 8: deduplicate
                key = (subject_canonical, relation, obj_canonical)
                if key in seen:
                    print(f"   ⏭ Skipping: duplicate triple")
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
                print(f"   ✅ Added: ({subject_canonical}:{subject_type}) "
                      f"-[{relation}]-> ({obj_canonical}:{object_type})")

            except Exception as e:
                print(f"⚠️ Error processing triple {item}: {e}")
                continue

        print(f"\n✅ FINAL TRIPLES: {len(triples)}")
        return triples