"""
Graph Service — two query methods:
  1. query()     — unified structured query (neighbours / path / by-type / subgraph)
  2. query_nl()  — natural language → Cypher → results

Connection: keep-alive + auto-reconnect on stale connections
"""

import os
import re
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


NL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a Neo4j Cypher expert.

Graph schema:
- Nodes: (:Entity {{ name: string, type: string }})
- Types: MODEL | DATASET | METRIC | ORGANIZATION | TASK | CONCEPT
- Relationships: DEVELOPED_BY | PROPOSED_BY | USES | USED_IN | TRAINED_ON |
  EVALUATED_ON | APPLIED_TO | USED_FOR | PART_OF | SUPPORTS | RELATED_TO | MENTIONS

STRICT Rules:
- ALWAYS use WHERE toLower(e.name) = toLower("some name") for name matching
- NEVER use {{name: toLower("...")}} — this matches the literal lowercased string, not the node
- LIMIT 20 unless asked otherwise
- Return ONLY the Cypher query — no explanation, no markdown, no backticks

Correct examples:
  Q: Which models did Google Brain develop?
  A: MATCH (m:Entity)-[r:DEVELOPED_BY]->(o:Entity)
     WHERE m.type = "MODEL" AND toLower(o.name) = toLower("Google Brain")
     RETURN m.name AS model

  Q: How is BLEU related to Transformer?
  A: MATCH path = shortestPath((a:Entity)-[*..6]-(b:Entity))
     WHERE toLower(a.name) = toLower("BLEU") AND toLower(b.name) = toLower("Transformer")
     RETURN [n IN nodes(path) | n.name] AS nodes, [r IN relationships(path) | type(r)] AS relations

  Q: What concepts does the Transformer use?
  A: MATCH (m:Entity)-[r:USES]->(c:Entity)
     WHERE toLower(m.name) = toLower("Transformer") AND c.type = "CONCEPT"
     RETURN c.name AS concept

  Q: Show all metrics evaluated on any model
  A: MATCH (model:Entity {{type: "MODEL"}})<-[r:EVALUATED_ON]-(metric:Entity {{type: "METRIC"}})
     RETURN model.name AS model, metric.name AS metric
     LIMIT 20

Question: {question}

Cypher:
"""
)


class GraphService:

    def __init__(self, model_name: str = "llama3"):
        self._uri  = os.getenv("NEO4J_URI")
        self._auth = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        self.driver = self._create_driver()
        self.driver.verify_connectivity()
        print("✅ Neo4j connection successful")
        self.llm = OllamaLLM(model=model_name)

    # ─────────────────────────────────────────────────────────────────────────
    def _create_driver(self):
        """Create driver with keep-alive and connection pool settings."""
        return GraphDatabase.driver(
            self._uri,
            auth=self._auth,
            max_connection_lifetime=3600,       # recreate connections after 1hr
            max_connection_pool_size=10,
            connection_acquisition_timeout=60,
            keep_alive=True                     # keep TCP connection alive
        )

    def _session(self):
        """
        Return a session, auto-reconnecting if the driver has gone stale.
        Use this instead of self.driver.session() everywhere.
        """
        try:
            session = self.driver.session()
            return session
        except Exception:
            print("🔄 Neo4j connection stale — reconnecting...")
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = self._create_driver()
            return self.driver.session()

    def close(self):
        self.driver.close()

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED STRUCTURED QUERY
    # type: "neighbours" | "path" | "by_type" | "subgraph"
    # ─────────────────────────────────────────────────────────────────────────
    def query(self, query_type: str, **kwargs) -> dict:

        with self._session() as session:

            # ── neighbours ───────────────────────────────────────────────────
            if query_type == "neighbours":
                entity = kwargs["entity"]
                limit  = kwargs.get("limit", 10)
                rows = session.run("""
                    MATCH (e:Entity)-[r]-(n)
                    WHERE toLower(e.name) = toLower($entity)
                    RETURN e.name AS source, type(r) AS relation,
                           n.name AS target, n.type AS target_type
                    LIMIT $limit
                """, entity=entity, limit=limit).data()
                return {"type": "neighbours", "entity": entity,
                        "count": len(rows), "results": rows}

            # ── path ─────────────────────────────────────────────────────────
            elif query_type == "path":
                from_e = kwargs["from_entity"]
                to_e   = kwargs["to_entity"]
                rows = session.run("""
                    MATCH path = shortestPath((a:Entity)-[*..6]-(b:Entity))
                    WHERE toLower(a.name) = toLower($from_e)
                      AND toLower(b.name) = toLower($to_e)
                    RETURN [n IN nodes(path) | n.name]          AS nodes,
                           [n IN nodes(path) | n.type]          AS node_types,
                           [r IN relationships(path) | type(r)] AS relations,
                           length(path)                          AS hops
                """, from_e=from_e, to_e=to_e).data()

                if not rows:
                    return {"type": "path", "found": False,
                            "from": from_e, "to": to_e, "path": []}
                r = rows[0]
                steps = [
                    {"from": r["nodes"][i], "from_type": r["node_types"][i],
                     "relation": rel,
                     "to": r["nodes"][i+1], "to_type": r["node_types"][i+1]}
                    for i, rel in enumerate(r["relations"])
                ]
                return {"type": "path", "found": True,
                        "from": from_e, "to": to_e,
                        "hops": r["hops"], "path": steps}

            # ── by_type ───────────────────────────────────────────────────────
            elif query_type == "by_type":
                etype = kwargs["entity_type"].upper()
                limit = kwargs.get("limit", 30)
                rows = session.run("""
                    MATCH (e:Entity {type: $etype})-[r]-(n)
                    RETURN e.name AS source, type(r) AS relation,
                           n.name AS target, n.type AS target_type
                    ORDER BY e.name LIMIT $limit
                """, etype=etype, limit=limit).data()

                grouped: dict = {}
                for row in rows:
                    src = row["source"]
                    if src not in grouped:
                        grouped[src] = {"entity": src, "type": etype, "relations": []}
                    grouped[src]["relations"].append({
                        "relation": row["relation"],
                        "target": row["target"],
                        "target_type": row["target_type"],
                    })
                return {"type": "by_type", "entity_type": etype,
                        "count": len(grouped), "entities": list(grouped.values())}

            # ── subgraph ──────────────────────────────────────────────────────
            elif query_type == "subgraph":
                entity = kwargs["entity"]
                depth  = min(kwargs.get("depth", 2), 4)
                limit  = kwargs.get("limit", 50)
                rows = session.run("""
                    MATCH path = (e:Entity)-[*1..$depth]-(n)
                    WHERE toLower(e.name) = toLower($entity)
                    UNWIND relationships(path) AS r
                    WITH startNode(r) AS src, type(r) AS rel, endNode(r) AS tgt
                    RETURN DISTINCT src.name AS source, src.type AS source_type,
                           rel AS relation, tgt.name AS target, tgt.type AS target_type
                    LIMIT $limit
                """, entity=entity, depth=depth, limit=limit).data()

                nodes, edges = {}, []
                for row in rows:
                    for name, ntype in [(row["source"], row["source_type"]),
                                        (row["target"], row["target_type"])]:
                        if name not in nodes:
                            nodes[name] = {"id": name, "type": ntype or "Unknown"}
                    edges.append({"source": row["source"],
                                  "relation": row["relation"],
                                  "target": row["target"]})
                return {"type": "subgraph", "entity": entity, "depth": depth,
                        "node_count": len(nodes), "edge_count": len(edges),
                        "nodes": list(nodes.values()), "edges": edges}

            else:
                raise ValueError(f"Unknown query_type '{query_type}'. "
                                 f"Use: neighbours | path | by_type | subgraph")

    # ─────────────────────────────────────────────────────────────────────────
    # NATURAL LANGUAGE QUERY
    # ─────────────────────────────────────────────────────────────────────────
    def query_nl(self, question: str, limit: int = 20) -> dict:
        print(f"\n🗣️  NL QUERY: {question}")

        try:
            raw    = self.llm.invoke(NL_PROMPT.format(question=question))
            cypher = self._extract_cypher(str(raw))
            cypher = self._fix_cypher(cypher)
        except Exception as e:
            return {"question": question, "cypher": None,
                    "results": [], "count": 0, "error": f"LLM failed: {e}"}

        print(f"🔍 FINAL CYPHER:\n{cypher}")

        if re.search(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b",
                     cypher, re.IGNORECASE):
            return {"question": question, "cypher": cypher, "results": [], "count": 0,
                    "error": "Write operations are not allowed in queries."}

        try:
            with self._session() as session:
                rows = session.run(cypher).data()
            print(f"✅ NL query returned {len(rows)} results")
            return {"question": question, "cypher": cypher,
                    "results": rows[:limit], "count": len(rows), "error": None}
        except Exception as e:
            print(f"⚠️ Cypher execution failed: {e}")
            return {"question": question, "cypher": cypher,
                    "results": [], "count": 0, "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_cypher(self, text: str) -> str:
        text = re.sub(r"```(?:cypher|sql)?", "", text).replace("```", "").strip()
        match = re.search(r"(MATCH\b.*)", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def _fix_cypher(self, cypher: str) -> str:
        """
        Fix the most common LLM mistake:
          BAD:  (n:Entity {name: toLower("Google Brain")})
          GOOD: (n:Entity) WHERE toLower(n.name) = toLower("Google Brain")
        """
        bad_pattern = re.compile(
            r'(\w+:\w+)\s*\{name:\s*toLower\(["\'](\w[^"\']*)["\'"]\)\}',
            re.IGNORECASE
        )
        matches = bad_pattern.findall(cypher)

        if not matches:
            return cypher

        cypher = bad_pattern.sub(r'(\1)', cypher)

        conditions = []
        for node_decl, name_value in matches:
            alias = node_decl.split(":")[0].strip()
            conditions.append(f'toLower({alias}.name) = toLower("{name_value}")')

        where_clause = " AND ".join(conditions)

        if re.search(r'\bWHERE\b', cypher, re.IGNORECASE):
            cypher = re.sub(r'\bWHERE\b', f"WHERE {where_clause} AND ",
                            cypher, count=1, flags=re.IGNORECASE)
        else:
            cypher = re.sub(r'\bRETURN\b', f"WHERE {where_clause}\nRETURN",
                            cypher, count=1, flags=re.IGNORECASE)

        print(f"🔧 Fixed name-filter for: {[m[1] for m in matches]}")
        return cypher