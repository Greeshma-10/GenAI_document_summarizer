"""
Graph Builder Module

Converts extracted triples into Neo4j Knowledge Graph
"""

import os
from neo4j import GraphDatabase
from typing import List
from dotenv import load_dotenv

from v2.graph.schema import Triple

# Load environment variables
load_dotenv()


class GraphBuilder:

    def __init__(self):

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

    def close(self):
        self.driver.close()

    # -------------------------------------
    # CLEAR GRAPH (NEW FUNCTION)
    # -------------------------------------
    def clear_graph(self):
        """
        Delete all nodes and relationships from the graph
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # -------------------------------------
    # INSERT TRIPLES
    # -------------------------------------
    def insert_triples(self, triples: List[Triple]):
        """
        Insert triples into Neo4j
        """

        with self.driver.session() as session:

            for triple in triples:
                session.execute_write(
                    self._create_relationship,
                    triple.subject,
                    triple.relation,
                    triple.object
                )

    # -------------------------------------
    # CREATE RELATIONSHIP
    # -------------------------------------
    @staticmethod
    def _create_relationship(tx, subject, relation, object_):

        query = f"""
            MERGE (a:Entity {{name:$subject}})
            MERGE (b:Entity {{name:$object}})
            MERGE (a)-[:{relation}]->(b)
            SET a.type = COALESCE(a.type, "Unknown")
            SET b.type = COALESCE(b.type, "Unknown")
            """

        tx.run(query, subject=subject, object=object_)