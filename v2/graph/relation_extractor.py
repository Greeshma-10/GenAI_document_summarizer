"""
Relation Extraction Module - Final Version

Fixes:
✔ USED_IN and TRAINED_ON now in RELATION_TYPES (schema fix)
✔ Metric triples auto-flipped: METRIC→MODEL becomes MODEL→METRIC via EVALUATED_ON
✔ Relation upgrader covers all meaningful type pairs
✔ Robust JSON parsing
"""

import json
import re
from typing import List, Dict, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from v2.graph.schema import Triple, RELATION_TYPES, CATEGORY_TO_SCHEMA_TYPE


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

# ─────────────────────────────────────────────────────────────────────────────
# Triples where subject/object should be FLIPPED for correctness
# e.g. LLM says  BLEU -[EVALUATED_ON]-> Transformer
#      but it should be  Transformer -[EVALUATED_ON]-> BLEU
# ─────────────────────────────────────────────────────────────────────────────
FLIP_RULES: Dict[tuple, str] = {
    # (subject_type, object_type, relation) → flip and use this relation
    ("METRIC", "MODEL", "EVALUATED_ON"): "EVALUATED_ON",
    ("METRIC", "MODEL", "USED_FOR"):     "EVALUATED_ON",
    ("DATASET","MODEL", "TRAINED_ON"):   "TRAINED_ON",
}


class RelationExtractor:

    def __init__(self, model_name: str = "llama3"):
        self.llm = OllamaLLM(model=model_name)

        self.prompt = PromptTemplate(
            input_variables=["text", "entities", "relations"],
            template="""
You are a precise knowledge graph extraction system for scientific papers.

Extract factual relationships between the listed entities from the text.

Text:
{text}

Entities (Name — Type):
{entities}

Allowed Relationships:
{relations}

Relation guide — use the MOST SPECIFIC relation for the entity types:
- MODEL  → ORGANIZATION  : DEVELOPED_BY
- MODEL  → DATASET       : TRAINED_ON
- MODEL  → METRIC        : EVALUATED_ON
- MODEL  → TASK          : APPLIED_TO
- MODEL  → CONCEPT       : USES
- CONCEPT → MODEL        : USED_IN
- METRIC → MODEL or TASK : USED_FOR
- ORGANIZATION → MODEL   : PROPOSED_BY
- Use RELATED_TO ONLY when no specific relation fits

Rules:
- Only extract relationships explicitly stated or strongly implied in the text
- Do NOT invent relationships
- Do NOT use RELATED_TO when a specific relation fits
- Skip obvious/generic triples (e.g. Transformer RELATED_TO classification)
- Return a single valid JSON array, nothing else — no explanation, no markdown

Output:
[
  {{"subject": "Entity1", "relation": "RELATION", "object": "Entity2"}}
]
"""
        )

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
        """Upgrade weak RELATED_TO/MENTIONS to a specific relation where possible."""
        if relation in WEAK_RELATIONS:
            preferred = PREFERRED_RELATION.get((subject_type, object_type))
            if preferred:
                print(f"   ⬆️  Upgraded: {relation} → {preferred} ({subject_type}→{object_type})")
                return preferred
        return relation

    # ─────────────────────────────────────────────────────────────────────────
    def _maybe_flip(
        self,
        subject: str, relation: str, obj: str,
        subject_type: str, object_type: str
    ):
        """
        Flip subject/object when the LLM has the direction backwards.
        e.g.  BLEU -[EVALUATED_ON]-> Transformer
              becomes  Transformer -[EVALUATED_ON]-> BLEU
        Returns (subject, relation, object, subject_type, object_type)
        """
        flip_relation = FLIP_RULES.get((subject_type, object_type, relation))
        if flip_relation:
            print(f"   🔄 Flipped: ({subject}:{subject_type}) -[{relation}]-> "
                  f"({obj}:{object_type})  →  ({obj}:{object_type}) -[{flip_relation}]-> "
                  f"({subject}:{subject_type})")
            return obj, flip_relation, subject, object_type, subject_type
        return subject, relation, obj, subject_type, object_type

    # ─────────────────────────────────────────────────────────────────────────
    def extract_relations(self, text: str, entities: Dict[str, List[str]]) -> List[Triple]:

        if not entities:
            return []

        entity_type_map: Dict[str, str] = self._build_entity_type_map(entities)
        if not entity_type_map:
            print("⚠️ No entities to extract relations from.")
            return []

        normalized_to_canonical: Dict[str, str] = {
            self._normalize(k): k for k in entity_type_map
        }

        entities_with_types = "\n".join(
            f"  {name} — {etype}" for name, etype in entity_type_map.items()
        )

        formatted_prompt = self.prompt.format(
            text=text,
            entities=entities_with_types,
            relations=", ".join(RELATION_TYPES)
        )

        try:
            response = self.llm.invoke(formatted_prompt)
            response_text = str(response)
        except Exception as e:
            print(f"⚠️ LLM invocation failed: {e}")
            return []

        print("\n🔗 RELATION LLM RESPONSE:\n", response_text)

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
        seen = set()   # deduplicate triples
        triples: List[Triple] = []

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

                # Step 4: drop MODEL RELATED_TO TASK (too generic)
                if relation == "RELATED_TO" and subject_type == "MODEL" and object_type == "TASK":
                    print(f"   ⏭ Skipping: MODEL RELATED_TO TASK is too generic")
                    continue

                # Step 5: deduplicate
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