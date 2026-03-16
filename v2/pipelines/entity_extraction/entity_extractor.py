from langchain_core.prompts import ChatPromptTemplate
from v2.services.bedrock_service import invoke_llm


# Research prompt (your existing one)
research_prompt = ChatPromptTemplate.from_template(
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


# Academic prompt (more generic disciplines)
academic_prompt = ChatPromptTemplate.from_template(
"""
You are an academic document entity extraction system.

Extract important academic entities from the text.

Most entities in academic documents should go under **key_concepts**.
Use other categories only if clearly applicable.

Entity categories:

models: theoretical models, frameworks, or systems
datasets: datasets, study data, surveys, or experimental data
metrics: statistical measures, rates, scores, or indicators
organizations: universities, research institutes, companies, or agencies
tasks: research objectives, analyses, experiments, or investigations
key_concepts: important theories, ideas, terminology, or academic concepts

Return ONLY valid JSON.

JSON format:

{{
 "models": [],
 "datasets": [],
 "metrics": [],
 "organizations": [],
 "tasks": [],
 "key_concepts": []
}}

Academic Text:
{text}
"""
)
def extract_entities(text: str, mode: str = "research"):

    if mode == "academic":
        prompt = academic_prompt
    else:
        prompt = research_prompt

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