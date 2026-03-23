"""
v2/prompts/fact_verification.py

Fact verification prompt — same pattern as chunk.py
"""


def build_fact_verification_prompt(
    claim: str,
    graph_evidence: str,
    text_evidence: str
) -> str:
    return f"""Respond with ONLY valid JSON. No text before or after. Start with {{ end with }}.

{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIED",
  "reason": "string",
  "confidence": 0.0
}}

Instructions:
- verdict must be exactly one of: SUPPORTED, CONTRADICTED, UNVERIFIED
- SUPPORTED: evidence directly confirms the claim
- CONTRADICTED: evidence directly conflicts with the claim
- UNVERIFIED: insufficient evidence to confirm or deny
- reason: brief explanation referencing specific evidence
- confidence: 0.0 to 1.0

Examples:
- Claim: "Transformer developed by Google Brain"
  Graph: (Transformer)-[DEVELOPED_BY]->(Google Brain)
  -> SUPPORTED, confidence 1.0

- Claim: "Transformer developed by OpenAI"
  Graph: (Transformer)-[DEVELOPED_BY]->(Google Brain)
  -> CONTRADICTED, confidence 0.9

- Claim: "Model uses dropout rate of 0.1"
  Graph: no relevant triple, Text: no relevant passage
  -> UNVERIFIED, confidence 0.0

No markdown, no explanation outside JSON, no trailing commas.

Claim: {claim}

Graph Evidence:
{graph_evidence}

Text Evidence:
{text_evidence}

JSON response:"""


def build_claim_extraction_prompt(text: str, max_claims: int) -> str:
    return f"""Respond with ONLY a valid JSON array. No text before or after. Start with [ end with ].

["claim 1", "claim 2"]

Instructions:
- Extract factual claims about NAMED entities (models, organizations, datasets, metrics)
- Claims must be about relationships: developed by, uses, trained on, evaluated on, applied to
- DO NOT extract procedural claims ("the paper discusses...")
- DO NOT extract vague claims without specific named entities
- Each claim must be a simple subject-verb-object statement
- Maximum {max_claims} claims
- No markdown, no explanation, no trailing commas

Text:
\"\"\"{text}\"\"\"

JSON array response:"""