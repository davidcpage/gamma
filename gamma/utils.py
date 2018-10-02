import pydot
from IPython.display import display, HTML, SVG
from gamma.core import FuncCache, input_nodes, depths, path_iter, topological_sort
import os
from urllib.request import urlretrieve
import hashlib
import tarfile
import zipfile
from tqdm import tqdm
import re


################
# plotting
################


PALETTE = ('#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
           '#b3de69', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f', '#1f78b4',
           '#33a02c', '#e31a1c', '#ff7f00', '#4dddf8',
           '#e66493', '#b07b87',  '#4e90e3', '#dea05e', '#d0c281',
           '#f0e189', '#e9e8b1', '#e0eb71', '#bbd2a4', '#6ed641', '#57eb9c',
           '#3ca4d4', '#92d5e7', '#b15928')


class ColorMap(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.palette = PALETTE

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

    def html(self):
        s = ('<div style="margin:2px;width:100px;height:15px;'
             'display:inline-block;line-height:15px;'
             'background-color:{};border-radius:9px;border-style:solid;'
             'border-width:1px;">'
             '<div style="width:90%;margin:auto;font-size:9px;'
             'text-align:center;overflow:hidden;">{}</div></div>')
        return ''.join((s.format(color, name) for name, color in self.items()))

    _repr_html_ = html

COLORS = ColorMap()

def parent(path):
    return path[:-1]

def stub(path):
    return path[-1]
 
get_type = lambda a: a['type'] if isinstance(a, dict) else type(a)
get_params = lambda a: a['params'] if isinstance(a, dict) else {} #better logic here
type_name = lambda t: getattr(t, '__name__', t) 


def draw(graphs, legend=True, scale=1, sep='/', direction='LR', extra_edges=(), extra_nodes=(), **kwargs):
    if isinstance(graphs, dict): #single graph
        graphs = (graphs,)
    types, svgs = [], []
    for graph in graphs:
        if not (isinstance(graph, dict) and len(graph)): continue
        dot_graph = DotGraph(graph, scale=scale, sep=sep, direction=direction)
        dot_graph.edges += list(extra_edges)
        dot_graph.nodes += list(extra_nodes)
        types += dot_graph.types
        svgs.append(dot_graph.svg(**kwargs))
    if legend:
        display(HTML(ColorMap.html({t: COLORS[t] for t in types})))
    for svg in svgs:
        display(SVG(svg))

def prepare_graph(graph):
    graph = dict(topological_sort(graph)) #fix ordering so that legend displays in better order
    graph = {n: ({'type': type_name(get_type(a)), 'params': get_params(a)}, i) for (n, (a, i)) in graph.items()}
    height = max(depths(graph).values())
    size = max(len(graph)/height, (height-0.3))/2
    nodes = [(k, {'tooltip': '%s %.1000r' % (attr['type'], attr['params']), 'fillcolor': COLORS[attr['type']],
                 }) for k, (attr, i) in graph.items()] 
    edges = [(src, k, {}) for k, n in graph.items()
            for src in input_nodes(n)]
    types = [a['type'] for (a, i) in graph.values()]
    return nodes, edges, size, types

def make_pydot(nodes, edges, direction='LR', sep='/', **kwargs):
    def make_subgraph(path, parent_graph):
        subgraph = pydot.Cluster(
           sanitise('/'.join(path)), label=sanitise(stub(path)), 
           style='rounded, filled', fillcolor='#77777744')
        parent_graph.add_subgraph(subgraph)
        return subgraph
    
    #graphviz doesn't like nodes named 'graph'
    sanitise = lambda k: k + ' ' if k in {'graph', 'subgraph', 'digraph'} else str(k)
     
    subgraphs = FuncCache(lambda path: make_subgraph(path, subgraphs[parent(path)]))
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')

    for node, attr in nodes:
        path = tuple(path_iter(node, sep))
        subgraphs[parent(path)].add_node(
            pydot.Node(name=sanitise(node), label=sanitise(stub(path)), **attr))
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(sanitise(src), sanitise(dst), **attr))

    return g

class DotGraph():
    def __init__(self, graph, scale=1, sep='/', direction='LR'):
        self.nodes, self.edges, self.size, self.types = prepare_graph(graph)
        self.scale, self.sep, self.direction = scale, sep, direction

    def dot_graph(self, **kwargs):
        return make_pydot(self.nodes, self.edges, size=self.size*self.scale, direction=self.direction, sep=self.sep, **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')

    def _repr_svg_(self):
        return self.svg()

"""
class DotGraph():
    def __init__(self, dot_graph):
        self.dot_graph = dot_graph
        self.svg = dot_graph.create(format='svg').decode('utf-8')
    def _repr_svg_(self):
        return self.svg
"""



def get_file(origin, fname=None, cache_dir='~/.gamma'):
    fname = fname or hashlib.sha1(origin.encode('utf-8')).hexdigest()
    fpath = os.path.join(os.path.expanduser(cache_dir), fname)
    basedir = os.path.dirname(fpath)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    sfx = ''
    for suffix in ('.tar.gz', '.tgz', '.tar.bz', 'tar'):
        if origin.endswith(suffix) and not fpath.endswith(suffix):
            sfx = suffix
    if not os.path.exists(fpath+sfx):
        desc = 'Downloading from {origin}'.format(origin=origin)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=desc) as pbar:
            def reporthook(blocknum, bs, size):
                pbar.total = size
                pbar.update(blocknum * bs - pbar.n)
            urlretrieve(origin, fpath+sfx, reporthook=reporthook)
    if not os.path.exists(fpath):
        if tarfile.is_tarfile(fpath+sfx):
            with tarfile.open(fpath+sfx) as archive:
                archive.extractall(fpath)
        elif zipfile.is_zipfile(fpath+sfx):
            with zipfile.ZipFile(fpath+sfx) as archive:
                archive.extractall(fpath)
    return fpath

