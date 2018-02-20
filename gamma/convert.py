import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict
from .core import union, reindex
import onnx
from onnx import numpy_helper

def from_onnx_nodes(nodes):
  to_np = lambda a: numpy_helper.to_array(a) if isinstance(a, onnx.TensorProto) else a
  return {k: ({
    'label': k,
    'params': {a.name: to_np(onnx.helper.get_attribute_value(a)) for a in n.attribute},
    'type': n.op_type
  }, list(n.input)) for n in nodes for k in n.output}

def from_onnx(onnx_model):
    net = from_onnx_nodes(onnx_model.graph.node)
    constants = from_onnx_nodes([onnx.helper.make_node('Constant', inputs=[], 
                       outputs=[x.name], value=x) for x in onnx_model.graph.initializer])
    inputs = {x.name: ({'type': 'Input', 'label': x.name, 'params': MessageToDict(x)}, [])
              for x in onnx_model.graph.input}
    graph = union(inputs, constants, net)
    return reindex(graph, {k: i for (i, k) in enumerate(graph.keys())})
  
def to_onnx(graph, name='', initializer=None):
  from_np = lambda a: numpy_helper.from_array(a) if isinstance(a, np.ndarray) else a
  nodes = [onnx.helper.make_node(attr['type'], [str(i) for i in inputs], [str(n)],
                                 label=attr['label'],
                                 **{k: from_np(v) for (k,v) in attr['params'].items()})
           for (n, (attr, inputs)) in graph.items()
           if attr['type'] != 'Input']
  inputs = [ParseDict(a['params'], onnx.ValueInfoProto())
            for (a,_) in graph.values() if a['type'] == 'Input']
  outputs = []
  onnx_graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializer)
  return onnx.helper.make_model(onnx_graph)

def from_tflow(graph_def):
    import tensorflow as tf
    unwrap = lambda arg: tf.make_ndarray(arg.tensor) if arg.HasField('tensor') else MessageToDict(arg)
    graph = {n.name: ({'type': n.op, 'label': n.name, 'params':
                       {k: unwrap(v) for k, v in n.attr.items()}
                       }, [i.split('^', 1)[-1].split(':', 1)[0] for i in n.input])
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
