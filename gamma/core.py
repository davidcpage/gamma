from collections import defaultdict
from itertools import count, chain
from unification import unify, reify, var

################
# dict utils
################


def union(*dicts):
    return {k: v for d in dicts for (k, v) in d.items()}


def gather(items):
    res = defaultdict(list)
    for k, v in items:
        res[k].append(v)
    return dict(res)


def assoc(dictionary, updates, inplace=False):
    if not inplace:
        dictionary = dictionary.copy()
    for k, v in updates.items():
        dictionary[k] = v
    return dictionary


class cache(dict):
    def __init__(self, func=None):
        self.func = func

    def __missing__(self, key):
        self[key] = val = self.func(key)
        return val

      
#################
# graph utils
#################

def get_attr(node):
    return node[0]


def get_inputs(node):
    return node[1]


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
    

def restrict(graph, inputs, outputs):
    neighbours = lambda node: (n for n in get_inputs(graph[node])
                                 if n not in inputs) if node in graph else []
    return walk_nodes(neighbours, outputs)


def edges(graph):
    return [(src, dst, port) for dst, (attr, inputs) in graph.items()
            for port, src in enumerate(inputs)]


def in_edges(graph):
    return gather((e[1], e) for e in edges(graph))


def out_edges(graph):
    return gather((e[0], e) for e in edges(graph))


def neighbourhoods(graph):
    return gather((e[i], (e[1-i], e)) for e in edges(graph) for i in (0, 1))


def external_inputs(graph):
    return {s for (s, _, _) in edges(graph) if s not in graph.keys()}


def strip_inputs(graph):
    return {n: (attr, [i for i in inputs if i in graph.keys()])
            for n, (attr, inputs) in graph.items()}


def strip_by_type(graph, type_):
    remove = {k for k, (attr, inputs) in graph.items()
              if attr['type'] == type_}
    return {n: (attr, [i for i in inputs if i not in remove])
            for n, (attr, inputs) in graph.items() if n not in remove}


def reindex(graph, name_func):
    if isinstance(name_func, dict):
        d = name_func
        name_func = lambda x: d.get(x, x)
    return {name_func(node): (attr, [name_func(i) for i in inputs])
            for node, (attr, inputs) in graph.items()}


def relabel(graph, label_func):
    if isinstance(label_func, dict):
        d = label_func
        label_func = lambda x: d.get(x, x)
    return {n: (dict(attr, label=label_func(attr['label'])), inputs)
            for n, (attr, inputs) in graph.items()}

  

#####################
# pattern matching
#####################


failure = []


def extend(bindings, extensions):
    return (assoc(bindings, ext, len(extensions) is 1) for ext in extensions)


def edge_constraint(edge, in_edges, out_edges, bindings):
    src, dst, port = edge
    if dst in bindings and src in bindings:
        return [bindings] if ((bindings[src], bindings[dst], port)
                              in in_edges.get(bindings[dst], [])) else failure
    elif dst in bindings:
        return extend(bindings, [{src: src_node} for (src_node, _, p)
                                 in in_edges.get(bindings[dst], []) if p == port])
    elif src in bindings:
        return extend(bindings, [{dst: dst_node} for (_, dst_node, p)
                                 in out_edges.get(bindings[src], []) if p == port])
    else:
        raise Exception('unconstrained edge')


def unify_(x, y, ctxt):
    new_ctxt = unify(x, y, ctxt)
    return failure if new_ctxt is False else [new_ctxt]


def node_constraint(node, pattern_node, graph, bindings):
    if node in bindings:
        b = bindings[node]
        if b in graph:
            return unify_(get_attr(graph[b]), get_attr(pattern_node), bindings)
        return failure 
    else:
        return [assoc(b, {node: n}, True) for n, (attr, _) in graph.items()
                    for b in unify_(attr, get_attr(pattern_node), bindings)]
 

def plan_query(pattern):
    neighbours = neighbourhoods(pattern)
    nodes = list(pattern.keys())
    query = []
    starting_nodes = {nodes[-1]} #better logic here..!
    for node in walk_nodes(lambda node: (n for (n, _) in neighbours[node]), starting_nodes):
        if node in nodes:
            query.append(node)
        query.extend(edge for _, edge in neighbours[node] if edge not in query)
    return query


def find_matches(graph, pattern):
    ins, outs = in_edges(graph), out_edges(graph)  # compute 'indices'
    proposals = [{}]
    for step in plan_query(pattern):
        if isinstance(step, tuple):
            edge = step
            new_proposals = (edge_constraint(edge, ins, outs, p)
                             for p in proposals)
        else:
            node = step
            new_proposals = (node_constraint(
                node, pattern[node], graph, p) for p in proposals)
        proposals = chain(*new_proposals)
    return list(proposals)


def apply_rule(graph, rule):
    LHS, RHS = rule
    matches = find_matches(graph, LHS)
    # remove matched nodes except for inputs
    remove = {n for match in matches for k, n in match.items() if k in LHS}
    # generate names for nodes to be added to the graph
    IDs = filter(lambda key: key not in graph, count(1))
    add = [reify(reindex(RHS, union(dict(zip(RHS.keys(), IDs)), match)), match)
           for match in matches]
    return union({k: v for k, v in graph.items() if k not in remove}, *add)


