import numpy as np
from google.protobuf.json_format import ParseDict, MessageToDict
from .core import reindex
import onnx
from onnx import numpy_helper
from .protobuf import unwrap
from itertools import chain

def from_onnx(onnx_model):
    g = unwrap(onnx_model.graph)
    ext_inputs = ((label, params, 'Input', []) for (label, params) in g.get('input',[]))
    constants =  ((v.name, {'value': v}, 'Constant', []) for v in g.get('initializer',[]))
    net =        ((label, dict(n.get('attribute',())), n['op_type'], n.get('input', [])) 
                   for n in g['node'] for label in n['output'])
    return {label: ({'label': label, 'params': params, 'type': type}, inputs)
            for (label, params, type, inputs) in chain(ext_inputs, constants, net)}

def to_onnx(graph, name, outputs=[], initializer=None):
    from_np = lambda a: numpy_helper.from_array(a) if isinstance(a, np.ndarray) else a
    onnx_type = lambda t: onnx.TensorProto.DESCRIPTOR.enum_types_by_name['DataType'].values_by_name[t].number
    nodes = [onnx.helper.make_node(attr['type'], [str(i) for i in inputs], [str(n)],
                                   **{k: from_np(v) for (k,v) in attr['params'].items()})
             for (n, (attr, inputs)) in graph.items()
             if attr['type'] != 'Input']
    inputs = [onnx.helper.make_tensor_value_info(name=str(n),
                                                 elem_type=onnx_type(a['params']['type']),
                                                 shape=a['params']['shape'],
                                                 doc_string=a['params'].get('doc_string', ''))
              for (n,(a,_)) in graph.items() if a['type'] == 'Input']
    outputs = [onnx.helper.make_tensor_value_info(name=str(n),
                                                  elem_type=onnx_type(a['elem_type']),
                                                  shape=a['shape'],
                                                  doc_string=a.get('doc_string', ''))
               for (n,a) in outputs]
    onnx_graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializer)
    return onnx.helper.make_model(onnx_graph)

def from_tflow(graph_def):
    graph = {n['name']: ({'type': n['op'], 'label': n['name'], 'params': n.get('attr',{})}, 
                         [i.split('^', 1)[-1].split(':', 1)[0] for i in n.get('input', [])])
             for n in unwrap(graph_def.node)}   
    return reindex(graph, {k: i for (i, k) in enumerate(graph.keys())})


def to_tflow(graph):
    import tensorflow as tf
    name_lookup = lambda n: graph[n][0]['label'] if n in graph else str(n)
    wrap = lambda arg: ({'tensor': MessageToDict(tf.make_tensor_proto(arg))} 
             if isinstance(arg, np.ndarray) else arg)
    nodes = [{'name': attr['label'], 'op': attr['type'],
              'attr': {k: wrap(v) for (k, v) in attr['params'].items()},
              'input': [name_lookup(i) for i in inputs]}
             for name, (attr, inputs) in graph.items()]
    return ParseDict({'node': nodes, 'library': {}}, tf.GraphDef())
