"""
Graph Service 
"""

import os
import re

from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM

from v2.prompts.cypher import build_nl_to_cypher_prompt
from v2.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cypher write-operation guard
# ─────────────────────────────────────────────────────────────────────────────
_WRITE_OPS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b", re.IGNORECASE
)

# Patterns that indicate a malformed LLM-generated Cypher node pattern
# e.g. (n:Entity {name: toLower("foo")}) — name filter should be in WHERE
_BAD_NAME_FILTER = re.compile(
    r'\((\w+:\w+)\s*\{name:\s*toLower\(["\']([^"\']*)["\'"]\)\}\)',
    re.IGNORECASE,
)
# e.g. (n:Entity {type: "X"})) — extra closing paren
_EXTRA_PAREN = re.compile(r'\{[^}]+\}\s*\)\s*\)')
# e.g. (n:Entity {type: "X"} {name: ...}) — two separate property maps
_DOUBLE_BRACE = re.compile(r'(\{[^}]+\})\s*(\{[^}]+\})')


class GraphService:

    def __init__(self, model_name: str = "llama3"):
        self._uri    = os.getenv("NEO4J_URI")
        self._auth   = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        self.driver  = self._create_driver()
        self.driver.verify_connectivity()
        logger.info("Neo4j connection successful")
        self.llm = OllamaLLM(model=model_name)

    # ─────────────────────────────────────────────────────────────────────────
    def _create_driver(self):
        return GraphDatabase.driver(
            self._uri,
            auth=self._auth,
            max_connection_lifetime=3600,
            max_connection_pool_size=10,
            connection_acquisition_timeout=60,
            keep_alive=True,
        )

    def _session(self):
        try:
            return self.driver.session()
        except Exception:
            logger.warning("Neo4j connection stale — reconnecting")
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
    # ─────────────────────────────────────────────────────────────────────────
    def query(self, query_type: str, **kwargs) -> dict:

        with self._session() as session:

            if query_type == "neighbours":
                entity = kwargs["entity"]
                limit  = kwargs.get("limit", 10)
                rows   = session.run("""
                    MATCH (e:Entity)-[r]-(n)
                    WHERE toLower(e.name) = toLower($entity)
                    RETURN e.name AS source, type(r) AS relation,
                           n.name AS target, n.type AS target_type
                    LIMIT $limit
                """, entity=entity, limit=limit).data()
                logger.debug("Neighbours query | entity='%s' results=%d", entity, len(rows))
                return {"type": "neighbours", "entity": entity,
                        "count": len(rows), "results": rows}

            elif query_type == "path":
                from_e = kwargs["from_entity"]
                to_e   = kwargs["to_entity"]
                # Guard: shortestPath fails when start == end node
                rows = session.run("""
                    MATCH path = shortestPath((a:Entity)-[*..6]-(b:Entity))
                    WHERE toLower(a.name) = toLower($from_e)
                      AND toLower(b.name) = toLower($to_e)
                      AND id(a) <> id(b)
                    RETURN [n IN nodes(path) | n.name]          AS nodes,
                           [n IN nodes(path) | n.type]          AS node_types,
                           [r IN relationships(path) | type(r)] AS relations,
                           length(path)                          AS hops
                """, from_e=from_e, to_e=to_e).data()

                if not rows:
                    logger.debug("No path found | from='%s' to='%s'", from_e, to_e)
                    return {"type": "path", "found": False,
                            "from": from_e, "to": to_e, "path": []}

                r     = rows[0]
                steps = [
                    {"from":      r["nodes"][i],      "from_type": r["node_types"][i],
                     "relation":  rel,
                     "to":        r["nodes"][i + 1],  "to_type":   r["node_types"][i + 1]}
                    for i, rel in enumerate(r["relations"])
                ]
                logger.debug("Path found | from='%s' to='%s' hops=%d", from_e, to_e, r["hops"])
                return {"type": "path", "found": True,
                        "from": from_e, "to": to_e,
                        "hops": r["hops"], "path": steps}

            elif query_type == "by_type":
                etype = kwargs["entity_type"].upper()
                limit = kwargs.get("limit", 30)
                rows  = session.run("""
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
                        "relation":    row["relation"],
                        "target":      row["target"],
                        "target_type": row["target_type"],
                    })
                logger.debug("by_type query | type='%s' entities=%d", etype, len(grouped))
                return {"type": "by_type", "entity_type": etype,
                        "count": len(grouped), "entities": list(grouped.values())}

            elif query_type == "subgraph":
                entity = kwargs["entity"]
                depth  = min(kwargs.get("depth", 2), 4)
                limit  = kwargs.get("limit", 50)

                # Depth must be inlined — Neo4j does not allow $param in path length bounds
                cypher = f"""
                    MATCH path = (e:Entity)-[*1..{depth}]-(n)
                    WHERE toLower(e.name) = toLower($entity)
                    UNWIND relationships(path) AS r
                    WITH startNode(r) AS src, type(r) AS rel, endNode(r) AS tgt
                    RETURN DISTINCT src.name AS source, src.type AS source_type,
                           rel AS relation, tgt.name AS target, tgt.type AS target_type
                    LIMIT $limit
                """
                rows = session.run(cypher, entity=entity, limit=limit).data()

                nodes, edges = {}, []
                for row in rows:
                    for name, ntype in [(row["source"], row["source_type"]),
                                        (row["target"], row["target_type"])]:
                        if name not in nodes:
                            nodes[name] = {"id": name, "type": ntype or "Unknown"}
                    edges.append({"source":   row["source"],
                                  "relation": row["relation"],
                                  "target":   row["target"]})

                logger.debug("Subgraph | entity='%s' depth=%d nodes=%d edges=%d",
                             entity, depth, len(nodes), len(edges))
                return {"type": "subgraph", "entity": entity, "depth": depth,
                        "node_count": len(nodes), "edge_count": len(edges),
                        "nodes": list(nodes.values()), "edges": edges}

            else:
                raise ValueError(
                    "Unknown query_type '%s'. Use: neighbours | path | by_type | subgraph"
                    % query_type
                )

    # ─────────────────────────────────────────────────────────────────────────
    # NATURAL LANGUAGE QUERY
    # ─────────────────────────────────────────────────────────────────────────
    def query_nl(self, question: str, limit: int = 20) -> dict:
        logger.info("NL query: %s", question)

        try:
            prompt = build_nl_to_cypher_prompt(question)
            raw    = self.llm.invoke(prompt)
            cypher = self._extract_cypher(str(raw))
            cypher = self._fix_cypher(cypher)
        except Exception:
            logger.exception("LLM failed during NL query")
            return {"question": question, "cypher": None,
                    "results": [], "count": 0, "error": "LLM failed"}

        logger.debug("Generated Cypher: %s", cypher)

        if _WRITE_OPS.search(cypher):
            logger.warning("Write operation blocked in NL query | cypher=%s", cypher)
            return {"question": question, "cypher": cypher, "results": [], "count": 0,
                    "error": "Write operations are not allowed in NL queries"}

        try:
            with self._session() as session:
                rows = session.run(cypher).data()
            logger.info("NL query returned %d results", len(rows))
            return {"question": question, "cypher": cypher,
                    "results": rows[:limit], "count": len(rows), "error": None}
        except Exception:
            logger.exception("Cypher execution failed | cypher=%s", cypher)
            return {"question": question, "cypher": cypher,
                    "results": [], "count": 0, "error": "Cypher execution failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # CYPHER CLEANUP HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_cypher(self, text: str) -> str:
        """Strip markdown fences and extract the MATCH clause."""
        text  = re.sub(r"```(?:cypher|sql)?", "", text).replace("```", "").strip()
        match = re.search(r"(MATCH\b.*)", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def _fix_cypher(self, cypher: str) -> str:
        # Fix 1 — name filter in node pattern
        matches = _BAD_NAME_FILTER.findall(cypher)
        if matches:
            cypher     = _BAD_NAME_FILTER.sub(r'(\1)', cypher)
            conditions = [
                'toLower(%s.name) = toLower("%s")' % (decl.split(":")[0].strip(), val)
                for decl, val in matches
            ]
            where_clause = " AND ".join(conditions)
            if re.search(r'\bWHERE\b', cypher, re.IGNORECASE):
                cypher = re.sub(r'\bWHERE\b', "WHERE %s AND " % where_clause,
                                cypher, count=1, flags=re.IGNORECASE)
            else:
                cypher = re.sub(r'\bRETURN\b', "WHERE %s\nRETURN" % where_clause,
                                cypher, count=1, flags=re.IGNORECASE)
            logger.debug("Fixed name-filter Cypher for: %s", [m[1] for m in matches])

        # Fix 2 — extra closing paren: "})" → "}"  then re-close the node properly
        if _EXTRA_PAREN.search(cypher):
            cypher = re.sub(r'(\{[^}]+\})\s*\)\s*\)', r'\1)', cypher)
            logger.debug("Fixed extra closing paren in Cypher")

        # Fix 3 — double property map: merge into one
        if _DOUBLE_BRACE.search(cypher):
            def _merge_braces(m):
                # Strip outer braces, join content, re-wrap
                left  = m.group(1)[1:-1].strip()
                right = m.group(2)[1:-1].strip()
                return "{%s, %s}" % (left, right)
            cypher = _DOUBLE_BRACE.sub(_merge_braces, cypher)
            logger.debug("Fixed double property map in Cypher")

        return cypher