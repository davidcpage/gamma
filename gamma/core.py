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


def values(container):
    if isinstance(container, dict):
        return container.values()
    return container


def items(container):
    if isinstance(container, dict):
        return container.items()
    return enumerate(container)


def map_values(func, container):
    if isinstance(container, dict):
        return {k: func(v) for (k, v) in container.items()}
    return [func(v) for v in container]


def filter_values(func, container):
    if isinstance(container, dict):
        return {k: v for (k, v) in container.items() if func(v)}
    return [v for v in container if func(v)]


class cache(dict):
    def __init__(self, func=None):
        self.func = func

    def __missing__(self, key):
        self[key] = val = self.func(key)
        return val

      
#################
# graph utils
#################


def get_inputs(node):
    return node[1]


def subgraph(graph, nodes):
    return {n: v for n, v in graph.items() if n in nodes}


def restrict(graph, inputs, outputs):
    nodes = []
    frontier = set(outputs)
    while frontier:
        node = frontier.pop()
        nodes.append(node)
        if node in graph:
            frontier.update(n for n in values(get_inputs(graph[node]))
                            if n not in inputs)
    return subgraph(graph, nodes)


def edges(graph):
    return [(src, dst, port) for dst, (attr, inputs) in graph.items()
            for port, src in items(inputs)]


def in_edges(graph):
    return gather((e[1], e) for e in edges(graph))


def out_edges(graph):
    return gather((e[0], e) for e in edges(graph))


def neighbourhoods(graph):
    return gather((e[i], (e[1-i], e)) for e in edges(graph) for i in (0, 1))


def inputs(graph):
    return {s for (s, _, _) in edges(graph) if s not in graph.keys()}


def strip_inputs(graph):
    f = set(graph.keys()).__contains__
    return {n: (attr, filter_values(f, inputs))
            for n, (attr, inputs) in graph.items()}


def strip_by_type(graph, type_):
    remove = {k for k, (attr, inputs) in graph.items()
              if attr['type'] == type_}
    return {n: (attr, filter_values(lambda x: x not in remove, inputs))
            for n, (attr, inputs) in graph.items() if n not in remove}


def reindex(graph, name_func):
    if isinstance(name_func, dict):
        d = name_func
        name_func = lambda x: d.get(x, x)
    return {name_func(node): (attr, map_values(name_func, inputs))
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


success = [{}]
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


def node_constraint(node, pattern_node, graph, bindings):
    if node in bindings:
        b = bindings[node]
        if b not in graph:
            raise KeyError(
                'Node %s is missing from the graph. Perhaps you need to supply it as an input?' % b)
        bindings = unify(graph[b][0], pattern_node[0], bindings)
        if bindings is False:
            return failure
        else:
            return [bindings]
    else:
        new_bindings = ((n, unify(attr, pattern_node[0], bindings))
                        for n, (attr, inputs) in graph.items())
        return [assoc(b, {node: n}, True) for (n, b) in new_bindings
                if b is not False]


def plan_query(pattern):
    starting_node = list(pattern.keys())[-1]  # better logic here..!
    neighbours = neighbourhoods(pattern)
    query = []
    frontier = {starting_node}
    while frontier:
        node = frontier.pop()
        if node in pattern.keys():
            query.append(node)
        for n, edge in neighbours[node]:
            if edge not in query:
                query.append(edge)
            if n not in query:
                frontier.add(n)
    return query


def find_matches(graph, pattern, unify_params=True):
    ins, outs = in_edges(graph), out_edges(graph)  # compute 'indices'
    proposals = success
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
    matches = find_matches(graph, LHS, True)
    # remove matched nodes except for inputs
    remove = {n for match in matches for k, n in match.items() if k in LHS}
    # generate names for nodes to be added to the graph
    IDs = filter(lambda key: key not in graph, ('v%g' % i for i in count(1)))
    add = [reindex(RHS, union(dict(zip(RHS.keys(), IDs)), match))
           for match in matches]
    # unify_params
    add = [reify(*p) for p in zip(add, matches)]
    return union({k: v for k, v in graph.items() if k not in remove}, *add)


