from v2.graph.graph_builder import GraphBuilder
from v2.graph.schema import Triple

triples = [
    Triple(subject="BERT", relation="DEVELOPED_BY", object="Google"),
    Triple(subject="BERT", relation="USES", object="Transformer")
]

graph = GraphBuilder()

graph.insert_triples(triples)

print("Triples inserted successfully!")

graph.close()