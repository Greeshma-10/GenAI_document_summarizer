"""
Relation Extraction Module

Converts:
Text + Extracted Entities → Knowledge Graph Triples
"""

import json
import re
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from v2.graph.schema import Triple, RELATION_TYPES


class RelationExtractor:
    """
    Extract relationships between entities using an LLM.
    """

    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the LLM model via Ollama.
        """

        self.llm = OllamaLLM(model=model_name)

        self.prompt = PromptTemplate(
            input_variables=["text", "entities", "relations"],
            template="""
You are a knowledge graph extraction system.

Identify meaningful relationships between entities in the text.

Text:
{text}

Entities:
{entities}

Allowed Relationships:
{relations}

Rules:
- Extract conceptual relationships such as USED_IN, APPLIED_TO, ENABLES, SUPPORTS, PART_OF.
- Use the closest allowed relationship if the exact one is unavailable.
- If two entities appear in the same sentence describing a relationship, extract a triple.

Return JSON list:

[
 {{"subject": "...", "relation": "...", "object": "..."}}
]
"""
        )

    def _clean_json(self, text: str):
        """
        Extract JSON array from LLM output.
        """

        match = re.search(r"\[.*\]", text, re.DOTALL)

        if not match:
            return None

        return match.group()

    def extract_relations(self, text: str, entities: List[str]) -> List[Triple]:
        """
        Extract relations between entities using the LLM.
        """

        if not entities:
            return []

        formatted_prompt = self.prompt.format(
            text=text,
            entities=", ".join(entities),
            relations=", ".join(RELATION_TYPES)
        )

        response = self.llm.invoke(formatted_prompt)

        response_text = str(response)

        json_text = self._clean_json(response_text)

        if not json_text:
            print("No JSON detected in LLM response")
            print(response_text)
            return []

        try:
            triples_json = json.loads(json_text)

        except Exception:
            print("Failed to parse relation output")
            print(response_text)
            return []

        triples: List[Triple] = []

        for item in triples_json:
            try:
                triple = Triple(
                    subject=item["subject"],
                    relation=item["relation"],
                    object=item["object"]
                )

                triples.append(triple)

            except Exception:
                continue

        return triples