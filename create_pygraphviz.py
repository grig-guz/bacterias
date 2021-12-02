import networkx as nx
import itertools
from collections import Counter


def edge_in_com(nodes, graph):
    edges = []
    for (i, j) in itertools.combinations(nodes, 2):
        if (i, j) in graph.edges():
            edges.append((i, j))
    return edges


karate = nx.generators.social.karate_club_graph()
karate_agr = nx.nx_agraph.to_agraph(karate)
print(karate_agr)
karate_agr.graph_attr['dpi'] = 180
#karate_agr.edge_attr.update(
 #   dir='both', arrowhead='inv', arrowtail='inv', penwidth=2.0)

karate_agr.node_attr.update(
    style='wedged',
    fontcolor='white',
    shape='circle',
    color='transparent',
    gradientangle=90)

colors = ['grey', 'pink', 'blue', 'purple']
communities = list(nx.community.asyn_fluidc(karate, 4))

most_edges = []
for n, com in enumerate(communities):
    edges = edge_in_com(com, karate)
    most_edges.extend(edges)
    for edge in edges:
        e = karate_agr.get_edge(*edge)
        e.attr['color'] = colors[n]
    for node in com:
        node = karate_agr.get_node(node)
        node.attr['fillcolor'] = colors[n]

other = [e for e in karate.edges() if e not in most_edges]

for edge in other:
    gn = karate_agr.get_node
    color = gn(edge[0]).attr['fillcolor']
    karate_agr.get_edge(*edge).attr['color'] = color

for n in karate_agr.nodes():
    cls = [e.attr['color'] for e in karate_agr.in_edges(n)]
    cls2 = [e.attr['color'] for e in karate_agr.out_edges(n)]
    cls = set(cls + cls2)
    if len(cls) > 1:
        # if n.attr['fillcolor'] != cls[0]:
        color1 = cls.pop()
        color2 = cls.pop()
        color_mix = ''.join([color1, ';', '0.33:', color2, ';', '0.34:', "#000000"])
        n.attr['fillcolor'] = color_mix

karate_agr.draw('karate.png', prog='dot')
