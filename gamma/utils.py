from pydot import Dot, Cluster, Node, Edge
from IPython.display import display, SVG, HTML
from gamma.core import FuncCache, get_inputs

################
# plotting
################


palette = ('#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
           '#b3de69', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f', '#1f78b4',
           '#33a02c', '#e31a1c', '#ff7f00', '#4dddf8',
           '#e66493', '#b07b87', '#f7397b', '#4e90e3', '#dea05e', '#d0c281',
           '#f0e189', '#e9e8b1', '#e0eb71', '#bbd2a4', '#6ed641', '#57eb9c',
           '#3ca4d4', '#92d5e7', '#b15928')


class ColorMap(dict):
    def __init__(self, palette=palette):
        self.palette = palette

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

    def _repr_html_(self):
        s = ('<div style="margin:2px;width:100px;height:15px;'
             'display:inline-block;line-height:15px;'
             'background-color:{};border-radius:9px;border-style:solid;'
             'border-width:1px;">'
             '<div style="width:90%;margin:auto;font-size:9px;'
             'text-align:center;overflow:hidden;">{}</div></div>')
        return ''.join((s.format(color, name) for name, color in self.items()))


COLORS = ColorMap()


def split(path):
    i = path.rfind('/')
    return path[:max(i, 0)], path[i+1:]


def parent(path):
    return split(path)[0]


def heights(graph):
    self = FuncCache()
    self.func = lambda node: 1 + max((self[n] for n in
                                      (get_inputs(graph[node]) if node in graph else ())),
                                     default=0)
    return {n: self[n] for n in graph}


def draw(graph, subgraphs=None, legend=True, scale=1, **kwargs):
    height = max(heights(graph).values())
    size = max(len(graph)/height, (height-0.3))*scale/1.5

    nodes = [(node, str(attr['label']),
              {'tooltip': '%s %s %r' % (node, attr['type'], attr['params']),
               'fillcolor': COLORS[attr['type']],
               }) for node, (attr, _) in graph.items()]
    edges = ((src, dst, {}) for dst, (_, inputs) in graph.items()
             for port, src in enumerate(inputs))
    g = draw_pydot(nodes, edges, subgraphs=subgraphs, size=size, **kwargs)
    if legend:
        types = {attr['type'] for _, (attr, _) in graph.items()}
        display(HTML(ColorMap._repr_html_({t: COLORS[t] for t in types})))
    display(SVG(g))


def draw_pydot(nodes, edges, direction='LR', **kwargs):
    def make_subgraph(path, parent_graph):
        subgraph = Cluster(
           path, label=split(path)[1], style='rounded, filled', fillcolor='#77777744')
        parent_graph.add_subgraph(subgraph)
        return subgraph

    subgraphs = FuncCache(lambda path: make_subgraph(
        path, subgraphs[parent(path)]))
    subgraphs[''] = g = Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')

    for node, path, attr in nodes:
        p, stub = split(path)
        subgraphs[p].add_node(Node(name=node, label=stub, **attr))
    for src, dst, attr in edges:
        g.add_edge(Edge(src, dst, **attr))

    return g.create_svg()
