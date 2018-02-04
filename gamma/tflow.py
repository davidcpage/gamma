  
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict, ParseDict
from gamma.core import reindex, map_values

def load_graph_def(pb_filename):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_filename, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
    return graph_def

def unwrap_node_attr(arg):
    if arg.HasField('tensor'):
        return tf.make_ndarray(arg.tensor)

    a = MessageToDict(arg)
    if 'shape' in a:
        return {'shape': [int(d['size']) for d in a['shape']['dim']]}
    return a


def wrap_node_attr(v):
    if isinstance(v, np.ndarray):
        return {'tensor': 
        MessageToDict(tf.make_tensor_proto(v))}
    if 'shape' in v:
        return {'shape': {'dim': [{'size': str(x)} for x in v['shape']]}}
    return v


def to_graph(graph_def):
    graph = {n.name: ({'type': n.op, 'label': n.name, 'params':
                       {k: unwrap_node_attr(v) for k, v in n.attr.items()}
                       }, [i.split(':', 1)[0] for i in n.input])
             for n in graph_def.node}
    return reindex(graph, {k: i for (i, k) in enumerate(graph.keys())})


def from_graph(graph):
    name_lookup = {n: attr['label'] for n, (attr, _) in graph.items()}
    nodes = [{'name': attr['label'], 'op': attr['type'],
              'attr': {k: wrap_node_attr(v) for (k, v) in attr['params'].items()},
              'input': map_values(name_lookup.get, inputs)}
             for name, (attr, inputs) in graph.items()]
    return ParseDict({'node': nodes, 'library': {}}, tf.GraphDef())


def build_graph_def(tf_code):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        tf_code()
    return tf_graph.as_graph_def()


def build_tf_graph(graph_def):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return tf_graph


def tf_run(graph_def, output_names, feeds):
    tf_graph = build_tf_graph(graph_def)

    def to_tensor(name): return tf_graph.get_tensor_by_name(name+':0')
    return tf.Session(graph=tf_graph).run([to_tensor(x) for x in output_names],
                                          feed_dict={to_tensor(k): v for (k, v) in feeds.items()})
