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

        # 🔍 DEBUG PRINTS
        print("\n🔗 DEBUG Neo4j Config:")
        print("URI:", uri)
        print("USER:", username)
        print("PASSWORD SET:", "YES" if password else "NO")

        if not uri:
            print("❌ ERROR: NEO4J_URI is not loaded from .env")

        # Initialize driver
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(username, password)
            )
            print("✅ Driver initialized")

        except Exception as e:
            print("❌ Driver init failed:", str(e))
            raise

        # 🔍 TEST CONNECTION
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
                print("✅ Neo4j connection successful")

        except Exception as e:
            print("❌ Connection failed:", str(e))

    def close(self):
        self.driver.close()

    # -------------------------------------
    # CLEAR GRAPH
    # -------------------------------------
    def clear_graph(self):
        """
        Delete all nodes and relationships from the graph
        """
        print("🧹 Clearing graph...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # -------------------------------------
    # INSERT TRIPLES
    # -------------------------------------
    def insert_triples(self, triples: List[Triple]):
        """
        Insert triples into Neo4j
        """

        print(f"📥 Inserting {len(triples)} triples...")

        with self.driver.session() as session:

            for i, triple in enumerate(triples):
                print(f"➡️ Triple {i+1}: {triple.subject} -[{triple.relation}]-> {triple.object}")

                session.execute_write(
                    self._create_relationship,
                    triple.subject,
                    triple.relation,
                    triple.object,
                    getattr(triple, "subject_type", "Unknown"),
                    getattr(triple, "object_type", "Unknown")
                )

    # -------------------------------------
    # CREATE RELATIONSHIP
    # -------------------------------------
    @staticmethod
    def _create_relationship(tx, subject, relation, object_, subject_type, object_type):

        # 🔒 Sanitize relation name (Neo4j safe)
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