from functools import singledispatch

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.internal.containers import MessageMap, RepeatedCompositeFieldContainer, RepeatedScalarFieldContainer
from google.protobuf.json_format import MessageToDict
from google.protobuf.pyext._message import RepeatedCompositeContainer, RepeatedScalarContainer

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2

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
    RepeatedCompositeContainer: lambda pb: [unwrap(v) for v in pb],
    RepeatedScalarContainer: lambda pb: [unwrap(v) for v in pb],
}

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


unwrap_onnx = {
    onnx.ModelProto: unwrap_standard,
    onnx.NodeProto: unwrap_standard,
    onnx.GraphProto: unwrap_standard,
    onnx.ValueInfoProto: lambda pb: (pb.name, MessageToDict(pb)), #too lazy to drill deeper into this for now
    #onnx.TypeProto: unwrap_standard,
    #onnx.TypeProto.Tensor: unwrap_standard,
    onnx.AttributeProto: lambda pb: (pb.name, unwrap(onnx.helper.get_attribute_value(pb))),
    onnx.TensorProto: lambda pb: (pb.name, onnx.numpy_helper.to_array(pb)),
}

unwrap = singledispatch(identity) #default is to do no unwrapping, making it easier to explore
for unwrappers in (unwrap_containers, unwrap_tf, unwrap_onnx):
    for (type_, func) in unwrappers.items():
        unwrap.register(type_, func)