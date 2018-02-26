from functools import singledispatch

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.internal.containers import MessageMap, RepeatedCompositeFieldContainer, RepeatedScalarFieldContainer
from google.protobuf.json_format import MessageToDict

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2
import numpy as np
import onnx

def identity(x):
    return x

def unwrap_standard(pb):
    return {f.name: unwrap(v) for (f, v) in pb.ListFields()}

def enum_to_string(field, value):
    if field.label == FieldDescriptor.LABEL_REPEATED:
        return [field.enum_type.values_by_number[v].name for v in value] 
    return field.enum_type.values_by_number[value].name    


unwrap_containers = {
    MessageMap: lambda pb: {k: unwrap(v) for k,v in pb.items()},
    RepeatedCompositeFieldContainer: lambda pb: [unwrap(v) for v in pb],
    RepeatedScalarFieldContainer: lambda pb: [unwrap(v) for v in pb],
}

# FIXME: test and remove once everyone is on google.protobuf >= 3.5 ??
try:
    from google.protobuf.pyext._message import RepeatedCompositeContainer, RepeatedScalarContainer, MessageMapContainer
    unwrap_containers[MessageMapContainer] = lambda pb: {k: unwrap(v) for k,v in pb.items()}
    unwrap_containers[RepeatedCompositeContainer] = lambda pb: [unwrap(v) for v in pb]
    unwrap_containers[RepeatedScalarContainer] = lambda pb: [unwrap(v) for v in pb]
except:
    pass
    

################
## tensorflow
###############


def unwrap_tf_AttrValue(pb):
    [(field, value)] = pb.ListFields()
    return enum_to_string(field, value) if field.type == FieldDescriptor.TYPE_ENUM else unwrap(value)

unwrap_tf = {
    tf.NodeDef: unwrap_standard,
    tf.GraphDef: unwrap_standard,
    tf.AttrValue: unwrap_tf_AttrValue,
    tf.AttrValue.ListValue: unwrap_tf_AttrValue,
    tensor_pb2.TensorProto: tf.make_ndarray,
    tensor_shape_pb2.TensorShapeProto: lambda pb: [x.size for x in pb.dim]
}

#############
## onnx
#############
class onnx_array(np.ndarray):
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', '')

def unwrap_onnx_TensorProto(pb):
    x = onnx.numpy_helper.to_array(pb).view(onnx_array)
    x.name = pb.name
    return x

def unwrap_onnx_ValueInfoProto(pb):
    if pb.type.HasField('tensor_type'):
        x = { 'elem_type': pb.type.tensor_type.elem_type,
              'shape':     [dim.dim_value for dim in pb.type.tensor_type.shape.dim] }
    else:
        # NOTE:  DNN-only implementations of ONNX MAY elect to not support non-tensor values
        #        as input and output to graphs and nodes. These types are needed to naturally
        #        support classical ML operators.  DNN operators SHOULD restrict their input
        #        and output types to tensors.
        x = MessageToDict(pb) # too lazy to drill deeper into this for now
    if pb.doc_string:
        x['doc_string'] = pb.doc_string
    return (pb.name, x)

unwrap_onnx = {
    onnx.ModelProto: unwrap_standard,
    onnx.NodeProto: unwrap_standard,
    onnx.GraphProto: unwrap_standard,
    onnx.ValueInfoProto: unwrap_onnx_ValueInfoProto,
    #onnx.TypeProto: unwrap_standard,
    #onnx.TypeProto.Tensor: unwrap_standard,
    onnx.AttributeProto: lambda pb: (pb.name, unwrap(onnx.helper.get_attribute_value(pb))),
    onnx.TensorProto: unwrap_onnx_TensorProto,
}

unwrap = singledispatch(identity) #default is to do no unwrapping, making it easier to explore
for unwrappers in (unwrap_containers, unwrap_tf, unwrap_onnx):
    for (type_, func) in unwrappers.items():
        unwrap.register(type_, func)