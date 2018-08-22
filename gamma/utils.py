import pydot
from IPython.display import display, HTML
from gamma.core import FuncCache, input_nodes, depths, path_iter
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

def draw(graphs, legend=True, scale=1, sep='/', **kwargs):
    if isinstance(graphs, dict): #single graph
        graphs = (graphs,)
    html, types, = '', []
    for graph in graphs:
        if not (isinstance(graph, dict) and len(graph)): continue
        type_name = lambda t: getattr(t, '__name__', t) 
        graph = {n: ({'type': type_name(get_type(a)), 'params': get_params(a)}, i) for (n, (a, i)) in graph.items()}
        height = max(depths(graph).values())
        size = max(len(graph)/height, (height-0.3))*scale/2
        nodes = [(k, tuple(path_iter(k, sep)),
                {'tooltip': '%s %s %.1000r' % (str(k), attr['type'], attr['params']),
                'fillcolor': COLORS[attr['type']],
                }) for k, (attr, i) in graph.items()]
        edges = ((src, k, {}) for k, n in graph.items()
                for src in input_nodes(n))
        svg = draw_pydot(nodes, edges, size=size, **kwargs)
        width = int(re.search('width="([0-9]*)pt"',svg).groups()[0])
        html += '<div style="min-width: {width}pt">{svg}</div>'.format(width=width, svg=svg)
        types += [a['type'] for (a, i) in graph.values()]
    if legend:
        html = ColorMap.html({t: COLORS[t] for t in types}) + html
    display(HTML(html))


def draw_pydot(nodes, edges, direction='LR', **kwargs):
    def make_subgraph(path, parent_graph):
        subgraph = pydot.Cluster(
           sanitise('/'.join(path)), label=sanitise(stub(path)), 
           style='rounded, filled', fillcolor='#77777744')
        parent_graph.add_subgraph(subgraph)
        return subgraph
    
    #graphviz doesn't like nodes named 'graph'
    sanitise = lambda k: k + ' ' if k in {'graph', 'subgraph', 'digraph'} else str(k)
     
    subgraphs = FuncCache(lambda path: make_subgraph(
        path, subgraphs[parent(path)]))
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')

    for node, path, attr in nodes:
        subgraphs[parent(path)].add_node(
            pydot.Node(name=sanitise(node), label=sanitise(stub(path)), **attr))
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(sanitise(src), sanitise(dst), **attr))

    return g.create(format='svg').decode('utf-8')



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

