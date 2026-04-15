from src.config import DATASET_TYPE
from src.pipeline import run_pipeline
from src.visualize import draw_graph

G, partition, texts = run_pipeline()

draw_graph(G, partition)

from collections import defaultdict

G, partition, texts = run_pipeline()

communities = defaultdict(list)

for node, comm in partition.items():
    communities[comm].append(texts[node])

print("\nTotal communities:", len(communities))

for comm, members in communities.items():
    print(f"\nCommunity {comm} ({len(members)} nodes):")
    for m in members[:3]:
        print("-", m[:80])

import json

with open(f"outputs/communities/{DATASET_TYPE}_communities.json", "w") as f:
    json.dump(communities, f, indent=2)

print("Dataset:", DATASET_TYPE)
print("Nodes:", len(texts))
print("Edges:", len(G.edges()))