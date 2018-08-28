import numpy as np
import mxnet
from mxnet import gluon, nd
from gamma.nodes import *
from gamma.core import bind_vars
from gamma.training import Transducer, Optimizer, transfer, add_, mul_, zeros_like, to_numpy

from collections import OrderedDict

class m_AddRelu(gluon.HybridBlock):
    def hybrid_forward(self, F, x, y):
        return F.relu(x+y)
    
class MaxPool(gluon.HybridBlock):
    def __init__(self, pool_size=(2, 2), strides=None, padding=0, layout='NCHW',
                 ceil_mode=False, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.op = gluon.nn.MaxPool2D(pool_size, strides, padding, layout='NCHW', 
                                                ceil_mode = ceil_mode, **kwargs)
        self.layout = layout
        
    def hybrid_forward(self, F, x):
        if self.layout == 'NHWC':
            return self.op(x.transpose([0,3,1,2])).transpose([0,2,3,1])
        return self.op(x)

class GlobalAvgPool(gluon.HybridBlock):
    def __init__(self, layout='NCHW', **kwargs):
        super(GlobalAvgPool, self).__init__(**kwargs)
        self.op = gluon.nn.GlobalAvgPool2D(layout='NCHW', **kwargs)
        self.layout = layout
    
    def hybrid_forward(self, F, x):
        if self.layout == 'NHWC':
            return self.op(x.transpose([0,3,1,2]))
        return self.op(x) 

class m_Add(gluon.HybridBlock):
    def hybrid_forward(self, F, x, y):
        return x + y

class m_ConcatPool(gluon.HybridBlock):
    def __init__(self, layout='NCHW', **kwargs):
        super().__init__(**kwargs)
        self.layout=layout
        self.avg_pool = gluon.nn.GlobalAvgPool2D()
        self.max_pool = gluon.nn.GlobalMaxPool2D()
        self.flatten = gluon.nn.Flatten()
        
    def hybrid_forward(self, F, x):
        if self.layout == 'NHWC':
            x = x.transpose([0,3,1,2])         
        return F.concat(
            self.flatten(self.avg_pool(x)),
            self.flatten(self.max_pool(x)),            
            dim=1
        )
    
class m_Correct(gluon.HybridBlock):
    def hybrid_forward(self, F, classifier, target):
        return F.argmax(classifier, axis=1).astype(np.int) == target
    

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
m_correct = node_def(m_Correct)


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

@bind_vars
def mxnet_correct(name, classifier, target):
    LHS = {name: (correct(), [classifier, target])}
    RHS = {name: (m_correct(), [classifier, target])}
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
    mxnet_relu(),
    mxnet_correct()
]

@transfer.register(nd.NDArray)
def _(data, device):
    return data.as_in_context(device)

@add_.register(nd.NDArray)
def _(x , a, y):
    if a is 0: return
    if a is 1: x[:] += y
    else: x[:] += a*y

@mul_.register(nd.NDArray)
def _(x, y):
    x[:] *=y

@zeros_like.register(nd.NDArray)
def _(x):
    return nd.zeros(x.shape, x.context, dtype=x.dtype)

@to_numpy.register(nd.NDArray)
def _(x): 
    return x.asnumpy() 


class MxnetGraph(gluon.Block):
    def __init__(self, graph):
        super().__init__()
        self.graph = list(topological_sort(graph))
        for n, (a, _) in self.graph: 
            if 'kwargs' in a['params']:
                del a['params']['kwargs']
            if issubclass(a['type'], gluon.Block):
                a['params']['prefix'] = n + '/'
            setattr(self, n, a['type'](**a['params']))

    def forward(self, inputs):
        cache = dict(inputs)
        for n, (a, i) in self.graph:
            cache[n] = getattr(self, n)(*[cache[x] for x in i])
        return cache
        
    def params_and_grads(self):
        return ((name, param.data(), param.grad()) for 
            (name, param) in self.collect_params().items() if param.grad_req != 'null')
  
    def zero_grad(self):
        pass

    def set_training(self, mode=True):
        return mxnet.autograd.set_training(mode)

    def recording_context(self):
        return mxnet.autograd.record()


class MxnetGraphHybrid(gluon.HybridBlock):
    def __init__(self, graph):
        super().__init__()
        self.graph = OrderedDict(topological_sort(graph))
        for n, (a, _) in self.graph.items(): 
            if 'kwargs' in a['params']:
                del a['params']['kwargs']
            if issubclass(a['type'], gluon.Block):
                a['params']['prefix'] = n + '/'
            setattr(self, n, a['type'](**a['params']))

    def hybrid_forward(self, F, x):
        cache = {'input': x}
        for n, (a, i) in self.graph.items():
            cache[n] = getattr(self, n)(*[cache[x] for x in i])
            if n == 'classifier': break
        return cache['classifier']
    
    def __call__(self, inputs):
        output_nodes = inputs.get('_output_nodes', ['loss', 'correct'])
        cache = inputs
        cache['classifier'] = self.forward(cache['input'])
        for node in output_nodes:
            (a, i) = self.graph[node]
            cache[node] = getattr(self, node)(*[cache[x] for x in i])
        return cache
  
    def params_and_grads(self):
        return ((name, param.data(), param.grad()) for 
            (name, param) in self.collect_params().items() if param.grad_req != 'null')
  
    def zero_grad(self):
        pass

    def set_training(self, mode=True):
        return mxnet.autograd.set_training(mode)

    def recording_context(self):
        return mxnet.autograd.record()
    
    
def to_nd(x, ctx=None):
    if isinstance(x, dict):
        return {k: to_nd(v, ctx) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    return nd.array(x, ctx=ctx, dtype=x.dtype)


def load_state(model, state_dict, ctx, strict=False):
    not_found = []
    for k, p in model.collect_params().items():
        if k in state_dict:
            p._load_init(to_nd(state_dict[k]), ctx=ctx)
        else:
            not_found.append(k)
            p.initialize(ctx=ctx)
    if len(not_found):
        warning = 'Warning initialising {not_found}'.format(not_found=not_found)
        if strict:
            raise Exception(warning)
        print(warning)
    return model
