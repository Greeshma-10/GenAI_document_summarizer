"""
v2/prompts/entity.py

Entity extraction prompt — same pattern as chunk.py
"""


def build_entity_extraction_prompt(text: str) -> str:
    return f"""Respond with ONLY valid JSON. No text before or after. Start with {{ end with }}.

{{
  "models": ["string"],
  "datasets": ["string"],
  "metrics": ["string"],
  "organizations": ["string"],
  "tasks": ["string"],
  "key_concepts": ["string"]
}}

Instructions:
- models: ML/AI model names (e.g. Transformer, BERT, GPT)
- datasets: dataset names (e.g. WMT 2014, ImageNet, Penn Treebank)
- metrics: evaluation metrics (e.g. BLEU, F1, accuracy, perplexity)
- organizations: research labs, universities, companies (e.g. Google Brain, MIT)
- tasks: NLP/ML tasks (e.g. machine translation, text classification)
- key_concepts: important technical concepts (e.g. self-attention, dropout)
- Only extract entities explicitly mentioned in the text
- Do NOT invent or hallucinate entities
- Return empty lists [] if no entities found for a category
- Entity names should be 1-5 words maximum
- No filler, no markdown, no trailing commas

Text:
\"\"\"{text}\"\"\"

JSON response:"""