from langchain_core.prompts import ChatPromptTemplate
from v2.services.bedrock_service import invoke_llm


prompt = ChatPromptTemplate.from_template(
"""
You are an expert research paper analyst.

Extract entities from the research text.

Entity categories:

models: AI or machine learning models
datasets: benchmark datasets
metrics: evaluation metrics (BLEU, F1, Accuracy)
organizations: companies, universities, labs
tasks: ML tasks (translation, classification, summarization)
key_concepts: technical ideas or algorithms

Return ONLY valid JSON in this format:

{{
 "models": [],
 "datasets": [],
 "metrics": [],
 "organizations": [],
 "tasks": [],
 "key_concepts": []
}}

Research Text:
{text}
"""
)


def extract_entities(text: str):

    formatted_prompt = prompt.format(text=text)

    try:
        entities = invoke_llm(formatted_prompt, max_gen_len=600)

        expected_keys = [
            "models",
            "datasets",
            "metrics",
            "organizations",
            "tasks",
            "key_concepts"
        ]

        for key in expected_keys:
            if key not in entities:
                entities[key] = []

        return entities

    except Exception as e:
        print("⚠️ Entity extraction failed:", e)

        return {
            "models": [],
            "datasets": [],
            "metrics": [],
            "organizations": [],
            "tasks": [],
            "key_concepts": []
        }