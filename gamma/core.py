from collections import defaultdict
from itertools import count, chain, islice

################
# logic
################

class var(object):
    _cache = {}
    def __new__(cls, token):
        try:
            return cls._cache[token]
        except KeyError:
            obj = object.__new__(cls)
            obj.token = token
            cls._cache[token] = obj
            return obj
    
    def __iter__(self):  
        #various parts of the code expect node['inputs'] to be an iterable.. not sure this is a good idea.
        return iter(())

    def __str__(self): return f'_{self.token}'
    __repr__ = __str__

    @classmethod
    def all(cls):
        return dict(enumerate(cls._cache))


def walk(key, d):
    while isinstance(key, var) and key in d:
        key = d[key]
    return key


class Wildcard():
    def __str__(self): return f'Wildcard_{id(self)}'
    __repr__ = __str__


def reify(x, s):
    if isinstance(x, var):
        return reify(s[x], s) if x in s else x
    elif isinstance(x, (tuple, list)):
        return type(x)(reify(xx, s) for xx in x)
    elif isinstance(x, dict):
        return {reify(k, s): reify(v, s) for k, v in x.items()}
    elif x is Wildcard:
        return var(x())
    return x


class UnificationError(Exception):
    pass


def _unify_inplace(u, v, s): #i.e. the bindings dict `s` gets updated in place
    u = walk(u, s)
    v = walk(v, s)
    #u and v could be vars, consts or (nested) datastructures of vars and consts
    if (u is v): return
    #numpy `==` is broken and breaks `== `for nested structure containing numpy arrays
    #so we have to test `==` in a try block...
    try:
        if u == v: return
    except ValueError: 
        pass    
    if isinstance(u, var): s[u] = v; return #occurs checks are missing
    if isinstance(v, var): s[v] = u; return
    if type(u) == type(v):
        if (isinstance(u, (list, tuple)) and len(u) == len(v)):
            for uu, vv in zip(u, v):  
                _unify_inplace(uu, vv, s)
            return
        elif (isinstance(u, dict) and u.keys() == v.keys()):
            for key, val in u.items():
                _unify_inplace(val, v[key], s)
            return    
    raise UnificationError


def unify(u, v, s=None):
    s = {} if s is None else s.copy()
    _unify_inplace(u, v, s)
    return s

################
# dict utils
################


def union(*dicts):
    return {k: v for d in dicts for (k, v) in d.items()}


def gather(items, reducer=set):
    res = defaultdict(list)
    for k, v in items:
        res[k].append(v)
    return {k: reducer(v) for k,v in res.items()}


def nested(flat_dict):
    res = {}
    for path, val in flat_dict.items():
        d = res
        for p in path[:-1]:
            d[p] = d = d.get(p, {})
        d[path[-1]] = val
    return res


def assoc(dictionary, key, val, inplace=False):
    dictionary = dictionary if inplace else dictionary.copy()
    dictionary[key] = val
    return dictionary


class FuncCache(dict):
    def __init__(self, func=None):
        self.func = func

    def __missing__(self, key):
        self[key] = val = self.func(key)
        return val

      
#################
# graph utils
#################

def subgraph(graph, nodes):
    return {n: a for n, a in graph.items() if n in nodes}


def walk_nodes(neighbours, starting_nodes):
    visited = set()
    frontier = set(starting_nodes)
    while frontier:
        node = frontier.pop()
        visited.add(node)
        frontier.update(n for n in neighbours(node) if n not in visited)
        yield node


def depths(graph):
    self = FuncCache()
    def depth(node):
        self[node] = 0 #avoid infinite recursion if graph contains cycles
        return 1 + max((self[n] for n in (input_nodes(graph[node]) if node in graph else ())), default=0)
    self.func = depth
    return {n: self[n] for n in graph}


def topological_sort(graph):
    return ((n, a) for _, _, (n, a) in sorted((depth, i, (node, graph[node])) 
            for i, (node, depth) in enumerate(depths(graph).items())))


def restrict(graph, inputs, outputs):
    neighbours = lambda node: (n for n in input_nodes(graph[node])
                                if n not in inputs) if node in graph else []
    return subgraph(graph, set(walk_nodes(neighbours, outputs)))


def input_nodes(node_attr):
    res = node_attr['inputs']
    return () if res is Wildcard else res
 

def edges(graph):
    return ((src, (dst_node, dst_port)) for dst_node, dst_attr in graph.items()
            for dst_port, src in enumerate(input_nodes(dst_attr))) 


def neighbourhoods(graph):
    edges_ = (((src, 'out'), (dst, dst_port)) for src, (dst, dst_port) in edges(graph))
    return nested(gather(((e[i], e[i-1]) for e in edges_ for i in [0,1])))


def external_inputs(graph, type_=set):
    return type_(s for node_attr in graph.values() for s in input_nodes(node_attr) if s not in graph.keys())
 

def strip(graph, nodes=external_inputs):
    #NB: this is primarily for display purposes e.g. if you want to remove external inputs 
    #or constants. Removing nodes in this way will typically break a computation graph.
    if callable(nodes):
        nodes = nodes(graph)
    return {n: dict(a, inputs=[i for i in input_nodes(a) if i not in nodes])
            for n, a in graph.items() if n not in nodes}


def truncate(graph, k):
    #truncate the graph to first k nodes by topological order - useful for inspecting/plotting
    return dict(islice(topological_sort(graph), k))


def gen_ids(reserved):
    return filter(lambda key: key not in reserved, count(1))
    

def reindex(graph, node_map=None):
    if node_map is None:
        node_map =  dict(zip((n for n, _ in topological_sort(graph)), gen_ids(reserved=external_inputs(graph))))
    node_map = new_node_ids(graph) if node_map is None else node_map
    f = lambda x: node_map.get(x, x)
    map_inputs = lambda inputs: (inputs if isinstance(inputs, var) else [f(i) for i in inputs])
    return {f(node): dict(attr, inputs=map_inputs(input_nodes(attr))) for node, attr in graph.items()}


def relabel(graph, label_func):
    if isinstance(label_func, dict):
        d = label_func
        label_func = lambda x: d.get(x, x)
    return {node: dict(attr, label=label_func(attr['label']))
            for node, attr in graph.items()}


def make_node_attr(type, params=None, label=None, inputs=None):
    params = {} if params is None else params
    inputs = [] if inputs is None else inputs
    return {'type': type, 'params': params, 'label': label, 'inputs': inputs}


def make_pattern(graph):
    return {var(n): make_node_attr(a['type'], var(f'{n}_params'), var(a['label']), 
             [var(x) for x in a['inputs']]) 
      for n, a in graph.items()}


def make_subgraph_node(subgraph, label, input_names=None):
    #we need to collect the set of all external_inputs of the subgraph
    #and make these inputs to the node. we rename inputs for internal use
    #in params['nodes] so that if reindex is called on the outer graph there 
    #is no need to rename things inside subgraph params['nodes']
    inputs = list(external_inputs(subgraph))
    input_names = input_names or (f'in{i}' for i in range(len(inputs)))
    return make_node_attr('Graph', {'nodes': reindex(subgraph, dict(zip(inputs, input_names))), 
                                        'input_names': input_names}, label, inputs)


def move_to_subgraphs(groups, graph):
    all_nodes = external_inputs(graph).union(graph)
    clashes = [group_name for group_name, _ in groups if group_name in all_nodes]
    if len(clashes):
        raise Exception(f'Name clash between groups {clashes} and nodes in the graph')
    subgraph_nodes = {group_name: make_subgraph_node({n: graph[n] for n in nodes}, group_name) 
                        for group_name, nodes in groups}
    remove = {n for _, nodes in groups for n in nodes}
    g = union({n: a for n, a in graph.items() if n not in remove}, subgraph_nodes)
    inputs = {n for a in g.values() for n in input_nodes(a)}
    ports = {n: make_node_attr('Port', {'node': n}, graph[n]['label'], [group_name]) for group_name, nodes in groups for n in nodes if n in inputs}  
    return union(g, ports)


def path_iter(label):
    #eg list(path_iter(('a/b', ('c', 'd/e'), 'f'))) == ['a', 'b', 'c', 'd', 'e', 'f']
    if isinstance(label, (list, tuple)):
        for l in label:
            yield from path_iter(l)
    else:
        yield from str(label).split('/')   


def path_str(label, sep='/'):
    return sep.join(path_iter(label))


def collapse(graph, levels=2):
    groups = gather((tuple(path_iter(a['label']))[:levels], n) for n, a in graph.items())
    groups = [(k, nodes) for k, nodes in groups.items() if len(nodes) > 1]
    return move_to_subgraphs(groups, graph)


#####################
# pattern matching
#####################

def plan_query(pattern, graph):
    #A query is a list of (LHS, ctxt->candidate RHS's)
    #see `search` for how this is used
    nbhds = {'graph': neighbourhoods(graph), 'pattern': neighbourhoods(pattern)}
    starting_node = list(pattern.keys())[-1] #better logic here..!
    query = [(starting_node, lambda ctxt: list(graph.keys()))]
    p_nbrs = lambda node: (n for nbrs in nbhds['pattern'].get(node, {}).values() 
                                  for (n, port) in nbrs)
    for node in walk_nodes(p_nbrs, {starting_node}):
        if node in pattern:
            #match node_attr
            query.append((pattern[node], lambda ctxt, node=node: [graph.get(ctxt[node], None)]))
        #match neighbouring nodes
        query.extend(
            (nbr, lambda ctxt, node=node, port=port: nbhds['graph'][ctxt[node]].get(port, ())) 
              for port, nbrs in nbhds['pattern'].get(node, {}).items() for nbr in nbrs)
    return query
    

def _match(LHS, candidates, ctxt):
    inplace = (len(candidates) == 1)
    for RHS in candidates:
        new_ctxt = ctxt if inplace else ctxt.copy()
        try:
            _unify_inplace(LHS, RHS, new_ctxt)
            yield new_ctxt
        except UnificationError:
            pass

def _search(pattern, graph):
    proposals = [{}]
    for (LHS, candidates) in plan_query(pattern, graph):
        proposals = chain(*(_match(LHS, candidates(ctxt), ctxt) for ctxt in proposals))
    return list(proposals)

def search(pattern, graph):
    pattern = reify(pattern, {}) #replace Wildcards with var(Wildcard())
    proposals = _search(pattern, graph)
    return [{k:v for (k, v) in proposal.items() if not isinstance(k.token, Wildcard)} for proposal in proposals]

def apply_rule(graph, rule):
    LHS, RHS = rule
    LHS = reify(LHS, {}) #replace Wildcards with var(Wildcard()); do it here so that we know which keys to remove below
    matches = _search(LHS, graph)
    # remove matched nodes except for inputs
    matched_nodes = {n for match in matches for n in reify(list(LHS.keys()), match)}
    ids = gen_ids(reserved=external_inputs(graph).union(graph))
    productions = [reify(RHS, union({k: next(ids) for k in RHS.keys()}, match)) for match in matches]
    return union({k: v for k, v in graph.items() if k not in matched_nodes}, *productions)

