def build_formatted_input(section_summaries):

    formatted = ""

    for section in section_summaries:
        formatted += f"""
SECTION {section['section_id']}
SUMMARY:
{section['section_summary']}

KEY POINTS:
{', '.join(section.get('section_key_points', []))}

RISKS:
{', '.join(section.get('section_risks_action_items', []))}
"""

    return formatted

## Research mode prompt
def build_research_executive_prompt(formatted_input: str) -> str:
    return f"""You must respond with ONLY valid JSON. Do not include any text before or after the JSON. Your response must start with {{ and end with }}

You are a senior enterprise strategy analyst creating a high-level executive brief.

Required JSON structure:
{{
  "executive_summary": "string",
  "executive_key_points": ["string"],
  "executive_risks_action_items": ["string"],
  "tldr": "string"
}}

Rules:
1. Output ONLY the JSON object
2. Do not wrap JSON in markdown code blocks
3. Do not add any explanatory text before or after
4. Merge duplicate themes across all sections
5. Abstract into high-level business insights
6. Summarize risks into broader strategic categories
7. executive_key_points: Maximum 5 strategic insights
8. executive_risks_action_items: Maximum 4 critical risks
9. tldr: Write ONE clear single-sentence summary of the entire document
10. Ensure all strings use double quotes
11. Ensure arrays use square brackets []
12. Do not use trailing commas
13. executive_summary MUST follow this four-part structure as flowing prose:
    - Problem Context: State the specific problem or gap being addressed — not generically, but as it actually appears in this document. Name the domain, the failure mode, or the unmet need directly.
    - Core Contribution: State what this work actually does or proposes — be concrete and specific. Avoid vague phrases like "novel approach" or "innovative framework." Say what it is.
    - Technical Architecture: Name the actual methods, models, components, or systems used. Do not describe them generically — pull the real terms from the input.
    - Strategic Impact: State the measurable or anticipated real-world outcome. Quantify where possible. Avoid corporate filler like "drives value" or "enables growth."
14. executive_summary must read like a sharp analyst briefing a skeptical executive — not like an academic abstract. Be direct, specific, and use the actual content of the document. No filler. No hedging. No textbook language.
15. Do NOT introduce any numerical performance improvements, KPIs, ROI figures, or quantitative business impact unless the exact figure is explicitly present in the section inputs provided.
16. Do NOT fabricate quantitative business impact. If the source data contains no metrics, the executive_summary and executive_key_points must reflect impact in qualitative terms only.
17. If you cannot find a specific metric in the input, write the insight without a number. A fabricated number is worse than no number — it is a governance failure.


Input data to synthesize:
{formatted_input}

JSON response:"""
    

## Academic mode prompt
def build_academic_executive_prompt(formatted_input: str) -> str:
    return f"""You must respond with ONLY valid JSON. Do not include any text before or after the JSON.
Your response must start with {{ and end with }}.

You are an expert academic tutor creating a precise study brief for a student.
Your goal is to make this material easy to understand — like a friend explaining 
it the night before an exam. Be plain, direct, and helpful.

Required JSON structure:
{{
  "executive_summary": "string",
  "executive_key_points": ["string"],
  "executive_risks_action_items": ["string"],
  "tldr": "string"
}}

Rules:
1.  Output ONLY the JSON object
2.  Do not wrap JSON in markdown code blocks
3.  Do not add any explanatory text before or after
4.  Ensure all strings use double quotes
5.  Ensure arrays use square brackets []
6.  Do not use trailing commas
7.  Do NOT fabricate metrics or statistics not present in the input
8.  Write for a student — plain English, no corporate language
9.  If the input is sparse or low-information, still return valid JSON with
    whatever can be reasonably extracted — never return an empty response

10. COVERAGE REQUIREMENT — This is your most critical rule:
    Before writing anything, mentally inventory every distinct concept, term, 
    framework, comparison, and mechanism present across ALL section summaries 
    in the input. Your output must account for ALL of them — either by naming 
    them directly, or by folding them into a point that explicitly captures them.
    Do NOT drop concepts just because they seem minor. If a section summary 
    mentioned it, your executive output must reflect it somewhere.

11. executive_summary: Write 4-6 sentences of plain English prose that naturally covers:
    - What topic or concept this material is about
    - What the central idea, mechanism, or framework is  
    - How it works — name the actual components, steps, or systems from the input
    - Why it matters or what it connects to
    - Weave in secondary concepts so they are not lost — do not save everything 
      for key_points. The summary must itself be informationally dense.
    IMPORTANT: If the input contains tabular data, comparisons, or numeric results,
    you MUST still produce a full prose executive_summary. Describe what is being
    compared, which models or configurations are involved, and what the results
    show in general terms — even if exact numbers are not present. Never leave
    executive_summary as an empty string or null when tabular or comparative
    content exists in the input.
    If the material is thin, write what you can honestly say without fabricating.

12. executive_key_points: Up to 5 key concepts — but prioritize COVERAGE over brevity:
    - Each point must capture one or more distinct concepts from the section inputs
    - If two concepts are closely related, combine them into one rich point rather 
      than dropping either
    - Use exact terms, definitions, comparisons, and named frameworks from the input
    - Each point must make sense on its own without reading the full text
    - Do NOT pad with vague filler — but do NOT leave out real concepts to stay brief
    - After writing all points, check: is every concept from the input represented 
      somewhere? If not, revise.

13. executive_risks_action_items: Up to 4 items — again, prioritize coverage:
    - Capture ALL misconceptions, limitations, prerequisites, and conceptual traps 
      present across the section inputs
    - If multiple limitations exist in the source, fold them into rich combined points
      rather than dropping any
    - If none exist in the input, return an empty array []
    - Do not invent risks, but do not discard real ones to hit a lower count

14. tldr: One plain sentence — what is this material about and what will a student
    understand after reading it. Write it like a friend, not a syllabus.
    The tldr must reflect the full scope of the material, not just the headline topic.

15. TABULAR AND COMPARATIVE DATA RULE:
    If any section summary mentions tables, comparisons, metrics, model evaluations,
    or numeric results — treat this as high-information content. You MUST:
    - Name the entities being compared (e.g. model names, configurations)
    - Name the metrics or dimensions used for comparison
    - Describe the general takeaway even if exact numbers are absent
    - Never skip or abbreviate the executive_summary because the source was a table
    A table is just data in a grid — summarize what it is comparing and why it matters.

16. NUMERICAL VALUES FROM TABULAR DATA:
    If the input contains tabular data with actual numerical values (e.g. accuracy scores,
    percentages, energy readings, benchmark numbers), you MUST use those exact numbers
    in your executive_summary and key_points — do not round, alter, or paraphrase them away.
    Rules for this:
    - Use the exact figures as they appear in the input (e.g. "BERT achieved 94.2% accuracy")
    - Do NOT generate, estimate, or invent any number that is not explicitly present in the input
    - Do NOT apply this rule to non-tabular text — only extract numbers that came from a table
    - If the table has many numbers, pick the most meaningful ones (best/worst performers,
      biggest differences) rather than listing every cell
    - If no numerical values are present in the tabular data, skip this rule entirely

17. SELF-CHECK before finalizing output:
    - Have you named every major concept from the section summaries?
    - Have you preserved every comparison or contrast present in the input?
    - Have you captured every limitation or risk mentioned in any section?
    - Is executive_summary a non-empty prose paragraph (4-6 sentences minimum)?
    - If the input had a table with numbers, have you used those exact numbers without modification?
    - If any answer is NO — revise your output before responding.

---
EXAMPLE — when input contains tabular or comparative data, your executive_summary must look like this:

Input contained: A table comparing BERT, RoBERTa, SVM, and Random Forest across accuracy, F1-score, and energy consumption on CPU, GPU, and edge devices.

Correct executive_summary:
"This material examines how four machine learning models — BERT, RoBERTa, SVM, and Random Forest — perform across different computational environments. The study evaluates them on standard classification metrics including accuracy, precision, recall, and F1-score, while also tracking energy consumption for each setup. The configurations tested range from CPU-only and GPU-enabled setups to distributed clusters and edge deployments, giving a broad picture of real-world trade-offs. Transformer-based models like BERT and RoBERTa tend to achieve stronger accuracy but at higher energy cost, while traditional models like SVM and Random Forest are more efficient but may sacrifice performance. Understanding these trade-offs is essential for choosing the right model when hardware or power constraints matter."

WRONG — never do this:
"executive_summary": ""
"executive_summary": "See key points below."
"executive_summary": "This section contains tabular data."


---

Input:
{formatted_input}

JSON response:"""