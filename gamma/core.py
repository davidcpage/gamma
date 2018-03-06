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

    def __str__(self):
        return "_" + str(self.token)
    __repr__ = __str__


def walk(key, d):
    while isinstance(key, var) and key in d:
        key = d[key]
    return key


def reify(x, s):
    if isinstance(x, var):
        return reify(s[x], s) if x in s else x
    elif isinstance(x, (tuple, list)):
        return type(x)(reify(xx, s) for xx in x)
    elif isinstance(x, dict):
        return {reify(k, s): reify(v, s) for k, v in x.items()}
    return x


class UnificationError(Exception):
    pass

class Wildcard():
    pass


def _unify_inplace(u, v, s): #i.e. the bindings dict `s` gets updated in place
    u = walk(u, s)
    v = walk(v, s)
    #u and v could be vars, consts or (nested) datastructures of vars and consts
    if (u is Wildcard or v is Wildcard): return #use type Wildcard as a wildcard. is this a good idea?
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

def reduce_(x, val):
    if isinstance(x, list): x.append(val)
    elif isinstance(x, set): x.add(val)
    elif isinstance(x, dict): x[val[0]] = val[1]
    elif isinstance(x, int): x = x + val
    return x

def gather(items, reducer=set):
    res = defaultdict(reducer)
    for k, v in items:
        res[k] = reduce_(res[k], v)
    return dict(res)

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
    return {n: v for n, v in graph.items() if n in nodes}

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
    self.func = lambda node: 1 + max((self[n] for n in
                                      (get_inputs(graph[node]) if node in graph else ())),
                                     default=0)
    return {n: self[n] for n in graph}


def topological_sort(graph):
    return (n for _, n in sorted((h, (k, graph[k])) 
            for (k, h) in depths(graph).items()))


def restrict(graph, inputs, outputs):
    neighbours = lambda node: (n for n in graph[node]['inputs']
                                if n not in inputs) if node in graph else []
    return subgraph(graph, set(walk_nodes(neighbours, outputs)))

def get_inputs(node):
    inputs = node['inputs']
    if isinstance(inputs, (list, tuple)):
        return inputs
    elif isinstance(inputs, var):
        return []
    raise Exception()

def edges(graph):
    return (((src, 'out'), (dst, port)) for dst, attr in graph.items()
            for port, src in enumerate(get_inputs(attr))) 

def neighbourhoods(graph):
    return nested(gather(((e[i], e[i-1]) for e in edges(graph) for i in [0,1])))


def external_inputs(graph):
    return {s for ((s, _), _) in edges(graph) if s not in graph.keys()}

def strip_nodes(graph, nodes):
    #NB: this is primarily for display purposes e.g. if you want to remove external inputs 
    #or constants. Removing nodes in this way will typically break a computation graph.
    return {n: dict(attr, inputs=[i for i in attr['inputs'] if i not in nodes])
            for n, attr in graph.items() if n not in nodes}


def truncate(graph, n):
    #truncate the graph to first n nodes by topological order - useful for inspecting/plotting
    return dict(islice(topological_sort(graph), n))

def reindex(graph, name_func):
    if isinstance(name_func, dict):
        d = name_func
        name_func = lambda x: d.get(x, x)
    return {name_func(k): dict(attr, inputs=attr['inputs'] if 
            isinstance(attr['inputs'], var) else 
            [name_func(i) for i in attr['inputs']]) for k, attr in graph.items()}


def relabel(graph, label_func):
    if isinstance(label_func, dict):
        d = label_func
        label_func = lambda x: d.get(x, x)
    return {k: dict(attr, label=label_func(attr['label']))
            for k, attr in graph.items()}

  
def make_node(type, params=None, label='', inputs=None):
    params = {} if params is None else params
    inputs = [] if inputs is None else inputs
    return {'type': type, 'params': params, 'label': label, 'inputs': inputs}


def make_pattern(graph):
    return {var(k): make_node(a['type'], var(f'{k}_params'), var(a['label']), 
             [var(x) for x in a['inputs']]) 
      for k, a in graph.items()}

#####################
# pattern matching
#####################

def plan_query(pattern, graph):
    graph_nbrs = neighbourhoods(graph)
    pattern_nbrs = neighbourhoods(pattern)
    starting_node = list(pattern.keys())[-1] #better logic here..!
    query = [(starting_node, lambda p: list(graph.keys()))]
    for node in walk_nodes(lambda node: (n for nbrs in pattern_nbrs.get(node,{}).values() 
        for (n, port) in nbrs), {starting_node}):
        if node in pattern:
            query.append((pattern[node], lambda p, node=node: (graph[p[node]],))) 
        query.extend(
            (nbr, lambda p, node=node, port=port: graph_nbrs[p[node]].get(port, ())) 
              for port, nbrs in pattern_nbrs.get(node,{}).items() for nbr in nbrs)
    return query
    
def _match(pattern, target, s):
    try:
        s = unify(pattern, target, s)
        return (s, )
    except UnificationError:
        return ()

def search(pattern, graph):
    proposals = [{}]
    for (pat, candidates) in plan_query(pattern, graph):
        proposals = chain(*(_match(pat, candidate, p) for p in proposals for candidate in candidates(p)))
    return list(proposals)


def apply_rule(graph, rule):
    LHS, RHS = rule
    matches = search(LHS, graph)
    # remove matched nodes except for inputs
    remove = {n for match in matches for k, n in match.items() if k in LHS}
    # generate names for nodes to be added to the graph
    IDs = filter(lambda key: key not in graph, count(1))
    add = [reify(reindex(RHS, dict(zip((k for k in RHS.keys() 
                if not isinstance(k, var)), IDs))), match)
                    for match in matches]
    return union({k: v for k, v in graph.items() if k not in remove}, *add)

