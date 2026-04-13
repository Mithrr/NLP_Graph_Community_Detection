import networkx as nx

def build_graph(sim_matrix, texts, threshold):
    G = nx.Graph()

    for i in range(len(texts)):
        G.add_node(i, text=texts[i])

    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if sim_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=sim_matrix[i][j])

    return G