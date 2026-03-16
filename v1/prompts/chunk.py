def build_chunk_summary_prompt(chunk_text: str, chunk_id: int) -> str:

    return f"""Respond with ONLY valid JSON. No text before or after. Start with {{ end with }}.

{{
  "chunk_id": {chunk_id},
  "summary": "string",
  "key_points": ["string"],
  "key_risks_action_items": ["string"]
}}

Instructions:
- summary: describe exactly what this chunk states
- key_points: grounded in the text; use exact numbers if present; if no numeric data write "No quantitative data present in this chunk"
- key_risks_action_items: explicit limitations only, no invented risks
- If tabular data is present: summarize structure, key metrics, trends, and numeric comparisons
- No filler, no markdown, no trailing commas

Text:
\"\"\"{chunk_text}\"\"\"

JSON response:"""