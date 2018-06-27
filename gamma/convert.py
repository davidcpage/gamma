from itertools import chain
import numpy as np
from google.protobuf.json_format import ParseDict, MessageToDict
import onnx
from onnx import numpy_helper
from .core import make_node_attr, path_iter, topological_sort
from .protobuf import unwrap

def make_tensor_value_info(name, elem_type, *args, **kwargs):
    elem_type = onnx.TensorProto.DESCRIPTOR.enum_types_by_name['DataType'].values_by_name[elem_type].number
    return onnx.helper.make_tensor_value_info(name, elem_type, *args, **kwargs)


def from_onnx(onnx_model):
    g = unwrap(onnx_model.graph)
    ext_inputs = (('Input', params, label, []) for (label, params) in g.get('input',[]))
    constants =  (('Constant', {'value': v}, v.name, []) for v in g.get('initializer',[]))
    net =        ((n['op_type'], dict(n.get('attribute',())), label, n.get('input', [])) 
                   for n in g['node'] for label in n['output'])
    return {l: make_node_attr(t, p, i) for (t, p, l, i) in chain(ext_inputs, constants, net)}


def to_onnx(graph, name, outputs=None, initializer=None):
    graph = list(topological_sort(graph))
    from_np = lambda a: numpy_helper.from_array(a) if isinstance(a, np.ndarray) else a
    nodes = [onnx.helper.make_node(a['type'], [str(x) for x in i], [str(n)],
                                   **{k: from_np(v) for (k,v) in a['params'].items()})
             for (n, (a, i)) in graph if a['type'] != 'Input']
    inputs = [make_tensor_value_info(str(n), **a['params'])
              for (n, (a, i)) in graph if a['type'] == 'Input']
    outputs = [] if outputs is None else [make_tensor_value_info(str(n), **a)
               for (n, (a, i)) in outputs]
    onnx_graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializer)
    return onnx.helper.make_model(onnx_graph)


def from_tflow(graph_def):
    graph = {n['name']: make_node_attr(n['op'], n.get('attr', {}), 
                         [i.split('^', 1)[-1].split(':', 1)[0] for i in n.get('input', [])])
             for n in unwrap(graph_def.node)}  
    return graph
  
def _to_string(label):
    return '/'.join(path_iter(label))

def to_tflow(graph):
    import tensorflow as tf
    name_lookup = lambda n: _to_string(graph[n]['label']) if n in graph else str(n)
    wrap = lambda arg: ({'tensor': MessageToDict(tf.make_tensor_proto(arg))} 
             if isinstance(arg, np.ndarray) else arg)
    nodes = [{'name': _to_string(attr['label']), 'op': attr['type'],
              'attr': {k: wrap(v) for (k, v) in attr['params'].items()},
              'input': [name_lookup(i) for i in attr['inputs']]}
             for name, attr in graph.items()]
    return ParseDict({'node': nodes, 'library': {}}, tf.GraphDef())
