
"""
v2/prompts/cypher.py

Natural language to Cypher prompt — same pattern as chunk.py
"""


def build_nl_to_cypher_prompt(question: str) -> str:
    return f"""Respond with ONLY the Cypher query. No text before or after. No markdown, no backticks.

Graph schema:
- Nodes: (:Entity {{ name: string, type: string }})
- Types: MODEL | DATASET | METRIC | ORGANIZATION | TASK | CONCEPT
- Relationships: DEVELOPED_BY | PROPOSED_BY | USES | USED_IN | TRAINED_ON |
  EVALUATED_ON | APPLIED_TO | USED_FOR | PART_OF | OUTPERFORMS |
  COMPARED_TO | RELATED_TO | MENTIONS

STRICT Rules:
- ALWAYS use WHERE toLower(e.name) = toLower("some name") for name matching
- NEVER use {{name: toLower("...")}} — this matches the literal lowercased string
- LIMIT 20 unless asked otherwise

Correct examples:
  Q: Which models did Google Brain develop?
  A: MATCH (m:Entity)-[r:DEVELOPED_BY]->(o:Entity)
     WHERE m.type = "MODEL" AND toLower(o.name) = toLower("Google Brain")
     RETURN m.name AS model

  Q: How is BLEU related to Transformer?
  A: MATCH path = shortestPath((a:Entity)-[*..6]-(b:Entity))
     WHERE toLower(a.name) = toLower("BLEU") AND toLower(b.name) = toLower("Transformer")
     RETURN [n IN nodes(path) | n.name] AS nodes,
            [r IN relationships(path) | type(r)] AS relations

  Q: What concepts does the Transformer use?
  A: MATCH (m:Entity)-[r:USES]->(c:Entity)
     WHERE toLower(m.name) = toLower("Transformer") AND c.type = "CONCEPT"
     RETURN c.name AS concept

  Q: Show all metrics evaluated on any model
  A: MATCH (model:Entity {{type: "MODEL"}})<-[r:EVALUATED_ON]-(metric:Entity {{type: "METRIC"}})
     RETURN model.name AS model, metric.name AS metric
     LIMIT 20

Question: {question}

Cypher:"""