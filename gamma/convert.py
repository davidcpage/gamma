import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict
from .core import union, reindex

def from_onnx(onnx_model):
    import onnx
    net = {k: ({
        'label': k,
        'params': {a.name: MessageToDict(a) for a in n.attribute},
        'type': n.op_type},  
        list(n.input)) for n in onnx_model.graph.node for k in n.output}

    constants = {x.name: ({'type': 'Constant', 'label': x.name, 'params': {'value': onnx.numpy_helper.to_array(x)}},[])
                  for x in onnx_model.graph.initializer}
    inputs = {x.name: ({'type': 'Input', 'label': x.name, 'params': MessageToDict(x)}, []) 
                for x in onnx_model.graph.input if x.name not in constants}
    return union(net, inputs, constants)


def from_tflow(graph_def):
    import tensorflow as tf
    unwrap = lambda arg: tf.make_ndarray(arg.tensor) if arg.HasField('tensor') else MessageToDict(arg)
    graph = {n.name: ({'type': n.op, 'label': n.name, 'params':
                       {k: unwrap(v) for k, v in n.attr.items()}
                       }, [i.split(':', 1)[0] for i in n.input])
             for n in graph_def.node}
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
