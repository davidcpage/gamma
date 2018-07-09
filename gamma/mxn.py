import numpy as np
import mxnet
from mxnet import gluon

from gamma.pytorch import *


class m_AddRelu(gluon.Block):
    def forward(self, x, y):
        return mxnet.nd.relu(x+y)
    
m_add_relu = node_def(m_AddRelu)
m_conv = node_def(gluon.nn.Conv2D)
m_x_entropy = node_def(gluon.loss.SoftmaxCrossEntropyLoss)
m_global_avg_pool = node_def(gluon.nn.GlobalAvgPool2D)
m_max_pool=node_def(gluon.nn.MaxPool2D)
m_linear = node_def(gluon.nn.Dense)
m_activation_func = node_def(gluon.nn.Activation)
m_bn = node_def(gluon.nn.BatchNorm)


@bind_vars
def mxnet_conv(name, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, _in):
    LHS = {name: (conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), [_in])}
    RHS = {name: (m_conv(out_channels, kernel_size, stride, padding, dilation, groups, use_bias=bias, in_channels=in_channels), [_in])}
    return LHS, RHS

@bind_vars
def mxnet_max_pool(name, kernel_size, stride, padding, _in):
    LHS = {name: (max_pool(kernel_size, stride, padding), [_in])}
    RHS = {name: (m_max_pool(kernel_size, stride, padding), [_in])}
    return LHS, RHS


@bind_vars
def mxnet_global_avg_pool(name, _in):
    LHS = {name: (global_avg_pool(), [_in])}
    RHS = {name: (m_global_avg_pool(), [_in])}
    return LHS, RHS

@bind_vars
def mxnet_linear(name, in_channels, out_channels, bias, _in):
    LHS = {name: (linear(in_channels, out_channels, bias), [_in])}
    RHS = {name: (m_linear(out_channels, use_bias=bias, in_units=in_channels), [_in])}
    return LHS, RHS

@bind_vars
def mxnet_add_relu(name, _in1, _in2):
    LHS = {name: (add_relu(), [_in1, _in2])}
    RHS = {name: (m_add_relu(), [_in1, _in2])}
    return LHS, RHS

@bind_vars
def mxnet_bn(name, in_channels, eps, affine, _in):
    LHS = {name: (bn(in_channels, eps, affine=affine), [_in])}
    RHS = {name: (m_bn(epsilon=eps, center=affine, scale=affine, in_channels=in_channels), [_in])} 
    return LHS, RHS

@bind_vars
def mxnet_activation_func(act_func, act_func_name, name, _in):
    LHS = {name: (activation_func(act_func, ), [_in])}
    RHS = {name: (m_activation_func(act_func_name), [_in])}
    return LHS, RHS


@bind_vars
def mxnet_x_entropy(name, logits, target):
    LHS = {name: (x_entropy(), [logits, target])}
    RHS = {name: (m_x_entropy(), [logits, target])}
    return LHS, RHS

rules = [
    mxnet_conv(), 
    mxnet_max_pool(), 
    mxnet_global_avg_pool(), 
    mxnet_linear(), 
    mxnet_add_relu(), 
    mxnet_bn(), 
    mxnet_activation_func(act_func=F.relu, act_func_name='relu'), 
    mxnet_x_entropy()
]

class MxnetGraph(gluon.Block):
    def __init__(self, graph):
        super().__init__()
        self.graph = dict(topological_sort(graph))
        for n, (a, _) in self.graph.items(): 
            if 'kwargs' in a['params']:
                del a['params']['kwargs']
            if issubclass(a['type'], gluon.Block):
                a['params']['prefix'] = n + '/'
            setattr(self, n, a['type'](**a['params']))

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (a, i) in self.graph.items():
            #print(n)
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

def to_nd(x):
    if isinstance(x, dict):
        return {k: to_nd(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    return mxnet.nd.array(x)

def load_state(model, state_dict):
    for k, p in model.collect_params().items():
        p._load_init(to_nd(state_dict[k]), ctx=mxnet.cpu(0))
    return model