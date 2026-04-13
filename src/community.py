import community as community_louvain

def detect_communities(G):
    return community_louvain.best_partition(G)