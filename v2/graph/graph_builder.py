"""
Graph Builder Module

Converts extracted triples into Neo4j Knowledge Graph
"""

import os
from neo4j import GraphDatabase
from typing import List
from dotenv import load_dotenv

from v2.graph.schema import Triple
from v2.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


class GraphBuilder:

    def __init__(self):
        uri      = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        logger.debug("Neo4j config | URI=%s | USER=%s | PASSWORD=%s",
                     uri, username, "SET" if password else "NOT SET")

        if not uri:
            logger.error("NEO4J_URI is not set in environment")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info("Neo4j driver initialized")
        except Exception:
            logger.exception("Neo4j driver initialization failed")
            raise

        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection successful")
        except Exception:
            logger.exception("Neo4j connection test failed")

    def close(self):
        self.driver.close()

    # ─────────────────────────────────────────────────────────────────────────
    def clear_graph(self):
        logger.info("Clearing graph")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # ─────────────────────────────────────────────────────────────────────────
    def insert_triples(self, triples: List[Triple]):
        logger.info("Inserting %d triples into Neo4j", len(triples))

        with self.driver.session() as session:
            for i, triple in enumerate(triples):
                logger.debug("Triple %d: %s -[%s]-> %s",
                             i + 1, triple.subject, triple.relation, triple.object)
                session.execute_write(
                    self._create_relationship,
                    triple.subject,
                    triple.relation,
                    triple.object,
                    getattr(triple, "subject_type", "Unknown"),
                    getattr(triple, "object_type", "Unknown")
                )

        logger.info("Inserted %d triples into Neo4j", len(triples))

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _create_relationship(tx, subject, relation, object_, subject_type, object_type):
        safe_relation = (
            relation.upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

        query = f"""
            MERGE (a:Entity {{name:$subject}})
            MERGE (b:Entity {{name:$object}})
            MERGE (a)-[:{safe_relation}]->(b)
            SET a.type = $subject_type
            SET b.type = $object_type
        """

        tx.run(
            query,
            subject=subject,
            object=object_,
            subject_type=subject_type,
            object_type=object_type
        )