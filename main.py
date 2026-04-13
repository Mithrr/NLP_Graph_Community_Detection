from src.pipeline import run_pipeline
from src.visualize import draw_graph

G, partition, texts = run_pipeline()

draw_graph(G, partition)