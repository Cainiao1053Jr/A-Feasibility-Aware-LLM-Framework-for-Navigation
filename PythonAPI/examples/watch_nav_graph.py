import networkx as nx
import matplotlib.pyplot as plt

# 读取 GraphML
G = nx.read_graphml("nav_graph.graphml")     # 无向图；若导出的是有向用 read_graphml 一样读

# 画个小图（大图别这么干，会很慢）
plt.figure(figsize=(6,6))
pos = {n: (float(G.nodes[n].get('x', 0)), float(G.nodes[n].get('y', 0))) for n in G.nodes}
nx.draw(G, pos, node_size=10, width=0.5)
plt.show()

# 用边权做最短路（比如 length_m）
path = nx.shortest_path(G, source="0", target="10", weight="length_m")
dist = nx.path_weight(G, path, weight="length_m")
print("path:", path, "dist(m):", dist)
