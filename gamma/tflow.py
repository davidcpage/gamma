  
import numpy as np
import tensorflow as tf

def load_graph_def(pb_filename):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_filename, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
    return graph_def


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
