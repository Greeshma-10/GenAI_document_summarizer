def build_section_prompt(section_chunks, section_id):

    formatted_input = ""

    for chunk in section_chunks:
        formatted_input += f"""
CHUNK {chunk['chunk_id']}
Summary: {chunk.get('summary', '')}
Key Points: {', '.join(chunk.get('key_points', []))}
Risks: {', '.join(chunk.get('key_risks_action_items', []))}
"""

    prompt = f"""Respond with ONLY valid JSON. Start with {{ end with }}.

{{
  "section_id": {section_id},
  "section_summary": "string",
  "section_key_points": ["string"],
  "section_risks_action_items": ["string"]
}}

Instructions:
- section_summary: state what this section actually says — name real methods, metrics, findings. No filler like "this section discusses"
- section_key_points: concrete insights with exact numbers, model names, techniques where present; max 5; each point must stand alone
- section_risks_action_items: specific limitations, tradeoffs, or open questions from this section; max 4; if none, state unresolved gaps
- Merge duplicate ideas across chunks; abstract into themes
- Preserve all numeric values exactly — do not reframe, round, or generalize them
- Do not introduce any metric or claim not explicitly present in the input
- No markdown, no trailing commas, no explanatory text

Input:
{formatted_input}

JSON response:"""
    return prompt
