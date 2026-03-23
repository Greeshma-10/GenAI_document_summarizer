"""
v2/prompts/relation.py

Relation extraction prompt — same pattern as chunk.py
"""


def build_relation_extraction_prompt(text: str, entities_with_types: str) -> str:
    return f"""Respond with ONLY a valid JSON array. No text before or after. Start with [ end with ].

PREFERRED relations (use these when they fit):
- DEVELOPED_BY   : model/tool created by an organization or person
- PROPOSED_BY    : model/concept introduced by an organization
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

You MAY define a NEW relation type if none of the above fits.
Use UPPERCASE_WITH_UNDERSCORES format. Examples: INTRODUCES, REPLACES, EXTENDS

Special rules for tables and results:
- "Transformer (big) 28.4 BLEU EN-DE" -> subject: Transformer (big), relation: EVALUATED_ON, object: BLEU
- "X outperforms Y" -> subject: X, relation: OUTPERFORMS, object: Y
- Training data mentioned -> extract TRAINED_ON

Rules:
- Only extract relationships explicitly stated or strongly implied in the text
- Do NOT invent relationships
- Be specific — prefer EVALUATED_ON over RELATED_TO for metrics
- Skip generic or obvious triples
- No markdown, no explanation, no trailing commas

Entities (Name — Type):
{entities_with_types}

Text:
\"\"\"{text}\"\"\"

JSON array response:"""