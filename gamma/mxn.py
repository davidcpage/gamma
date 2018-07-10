import numpy as np
import mxnet
from mxnet import gluon

from gamma.pytorch import *


class m_AddRelu(gluon.Block):
    def forward(self, x, y):
        return mxnet.nd.relu(x+y)
    
class MaxPool(gluon.Block):
    def __init__(self, pool_size=(2, 2), strides=None, padding=0, layout='NCHW',
                 ceil_mode=False, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.op = gluon.nn.MaxPool2D(pool_size, strides, padding, layout='NCHW', 
                                                ceil_mode = ceil_mode, **kwargs)
        self.layout = layout
        
    def forward(self, x):
        if self.layout == 'NHWC':
            return self.op(x.transpose([0,3,1,2])).transpose([0,2,3,1])
        return self.op(x)

class GlobalAvgPool(gluon.Block):
    def __init__(self, layout='NCHW', **kwargs):
        super(GlobalAvgPool, self).__init__(**kwargs)
        self.op = gluon.nn.GlobalAvgPool2D(layout='NCHW', **kwargs)
        self.layout = layout
    
    def forward(self, x):
        if self.layout == 'NHWC':
            return self.op(x.transpose([0,3,1,2]))
        return self.op(x) 

class m_Add(gluon.Block):
    def __init__(self, inplace=True, **kwargs):
        super().__init__(**kwargs)
        self.inplace=inplace
    def forward(self, x, y):
        if self.inplace: return mxnet.nd.elemwise_add(x, y, out=x)
        else: return x + y

class m_ConcatPool(gluon.Block):
    def __init__(self, layout='NCHW', **kwargs):
        super().__init__(**kwargs)
        self.layout=layout
        self.avg_pool = gluon.nn.GlobalAvgPool2D()
        self.max_pool = gluon.nn.GlobalMaxPool2D()
        self.flatten = gluon.nn.Flatten()
        
    def forward(self, x):
        if self.layout == 'NHWC':
            x = x.transpose([0,3,1,2])         
        return mxnet.ndarray.concat(
            self.flatten(self.avg_pool(x)),
            self.flatten(self.max_pool(x)),            
            dim=1
        )
    

m_add = node_def(m_Add)
m_concat_pool = node_def(m_ConcatPool)
m_add_relu = node_def(m_AddRelu)
m_conv = node_def(gluon.nn.Conv2D)
m_x_entropy = node_def(gluon.loss.SoftmaxCrossEntropyLoss)
m_global_avg_pool = node_def(GlobalAvgPool)
m_max_pool=node_def(MaxPool)
m_linear = node_def(gluon.nn.Dense)
m_activation_func = node_def(gluon.nn.Activation)
m_bn = node_def(gluon.nn.BatchNorm)


@bind_vars
def mxnet_conv(name, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, layout, _in):
    LHS = {name: (conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), [_in])}
    RHS = {name: (m_conv(out_channels, kernel_size, stride, padding, dilation, groups, use_bias=bias, layout=layout,  in_channels=in_channels), [_in])}
    return LHS, RHS

@bind_vars
def mxnet_max_pool(name, kernel_size, stride, padding, layout, _in):
    LHS = {name: (max_pool(kernel_size, stride, padding), [_in])}
    RHS = {name: (m_max_pool(kernel_size, stride, padding, layout=layout), [_in])}
    return LHS, RHS



@bind_vars
def mxnet_global_avg_pool(name, layout, _in):
    LHS = {name: (global_avg_pool(), [_in])}
    RHS = {name: (m_global_avg_pool(layout=layout), [_in])}
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
def mxnet_bn(name, in_channels, eps, affine, axis, _in):
    LHS = {name: (bn(in_channels, eps, affine=affine), [_in])}
    RHS = {name: (m_bn(epsilon=eps, center=affine, scale=affine, in_channels=in_channels, axis=axis), [_in])} 
    return LHS, RHS

@bind_vars
def mxnet_activation_func(act_func, act_func_name, name, _in):
    LHS = {name: (activation_func(act_func, ), [_in])}
    RHS = {name: (m_activation_func(act_func_name), [_in])}
    return LHS, RHS


@bind_vars
def mxnet_concat_pool(name, layout, _in):
    LHS = {name: (pool(), [_in])}
    RHS = {name: (m_concat_pool(layout=layout), [_in])}
    return LHS, RHS

@bind_vars
def mxnet_add(name, x, y):
    LHS = {name: (add(), [x, y])}
    RHS = {name: (m_add(), [x, y])}
    return LHS, RHS    
        
        
@bind_vars
def mxnet_relu(name, _in):
    LHS = {name: (relu(True), [_in])}
    RHS = {name: (m_activation_func('relu'), [_in])}
    return LHS, RHS

@bind_vars
def mxnet_x_entropy(name, logits, target):
    LHS = {name: (x_entropy(), [logits, target])}
    RHS = {name: (m_x_entropy(), [logits, target])}
    return LHS, RHS

def rules(layout='NCHW'):
    return [
    mxnet_conv(layout=layout), 
    mxnet_max_pool(layout=layout), 
    mxnet_global_avg_pool(layout=layout), 
    mxnet_linear(), 
    mxnet_add_relu(), 
    mxnet_bn(axis=layout.index('C')), 
    mxnet_activation_func(act_func=F.relu, act_func_name='relu'), 
    mxnet_x_entropy(),
    mxnet_concat_pool(layout=layout), 
    mxnet_add(), 
    mxnet_relu()
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

def to_nd(x, ctx):
    if isinstance(x, dict):
        return {k: to_nd(v, ctx) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    return mxnet.nd.array(x, ctx=ctx)

def load_state(model, state_dict, ctx):
    for k, p in model.collect_params().items():
        p._load_init(to_nd(state_dict[k], ctx), ctx=ctx)
    return model


