from v2.graph.relation_extractor import RelationExtractor

text = """
BERT was developed by Google using the Transformer architecture.
"""

entities = ["BERT", "Google", "Transformer"]

extractor = RelationExtractor()

triples = extractor.extract_relations(text, entities)

for t in triples:
    print(t)