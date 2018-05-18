import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
from gamma.core import *

class TorchGraph(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = dict(topological_sort(graph))
        for n, a in self.graph.items(): 
            setattr(self, n, a['type'](**a['params']))

    def forward(self, inputs):
        cache = dict(inputs)
        for n, a in self.graph.items():
            cache[n] = getattr(self, n)(*[cache[i] for i in a['inputs']])
        return cache

def load_state(net, state_dict):
    for key, val in state_dict.items():
        *head, tail = key.split('.')
        mod = getattr(net, '.'.join(head))
        #'https://download.pytorch.org/models/resnet152-b121ed2d.pth' stores parameters as tensors...
        if isinstance(getattr(mod, tail), nn.Parameter) and not isinstance(val, nn.Parameter):
            val = nn.Parameter(val) 
        setattr(mod, tail, val)    

def to_numpy(x): return x.detach().cpu().numpy()    

class Identity(nn.Module):
    def forward(self, x): return x
    
class Add(nn.Module):
    def __init__(self,  inplace):
        super().__init__()
        self.inplace = inplace
    def forward(self, x, y): 
        if self.inplace: return x.add_(y)
        else: return x + y
       
class GlobalAvgPool2d(nn.Module):
    def forward(self, x): return F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), x.size(1))
    
class ConvOp(nn.Module):
    def __init__(self,  stride, padding):
        super().__init__()
        self.stride, self.padding = stride, padding
    def forward(self, x, weight): return F.conv2d(x, weight, stride=self.stride, padding=self.padding)

class LinearOp(nn.Module):
    def forward(self, x, weight, bias): return F.linear(x, weight, bias)
        
class BatchNormOp(nn.Module):
    def __init__(self, training):
        super().__init__()
        self.training = training
    def forward(self, x, running_mean, running_var, weight, bias): return F.batch_norm( x, running_mean, running_var, weight, bias, training=self.training)

class ConcatPool2d(nn.Module):
    def forward(self, x):
        # pylint: disable=E1101
        return torch.cat([
            F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), x.size(1)),
            F.adaptive_max_pool2d(x, (1,1)).view(x.size(0), x.size(1)),
        ], 1)
        #pylint: enable=E1101

class Constant(nn.Module):
    def __init__(self, value, size=None):
        super().__init__()
        self.value = value
        self.size = tuple(value.size()) if size is None else size
    def forward(self): return self.value


node = namedtuple('node', ('type', 'param_names'))                
node.__call__ = lambda self, label, param_values, inputs=None: make_node_attr(
    self.type, dict(zip(self.param_names, param_values)), label, inputs)

identity  = node(Identity, [])    
pool      = node(ConcatPool2d, [])
linear    = node(nn.Linear, ['in_features', 'out_features'])
bn        = node(nn.BatchNorm2d, ['num_features'])
conv      = node(nn.Conv2d, ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias', 'groups'])
conv_op   = node(ConvOp, ['stride', 'padding'])
bn_op     = node(BatchNormOp, ['training'])
linear_op = node(LinearOp, [])
max_pool   = node(nn.MaxPool2d, ['kernel_size', 'stride', 'padding'])

global_avg_pool   = node(GlobalAvgPool2d, [])

relu      = node(nn.ReLU, ['inplace'])
x_entropy = node(nn.CrossEntropyLoss, [])
add       = node(Add, ['inplace'])
constant = node(Constant, ['value', 'size'])

##############
## Rules
##############

_in, _out, _0, _1, _2, _3 = var('in'), var('out'), *map(var, range(4))

@bind_vars
def expand_conv(conv_name, in_channels, out_channels, kernel_h, kernel_w, stride, padding):
    LHS = {_out: conv(conv_name, [in_channels, out_channels, (kernel_h, kernel_w), stride, padding, False], [_in])}
    RHS = {
        _0: constant((conv_name, 'weight'), [None, (out_channels, in_channels, kernel_h, kernel_w)], []),
        _out: conv_op((conv_name, 'out'), [stride, padding], [_in, _0])
    }
    return LHS, RHS

@bind_vars
def expand_linear(linear_name, in_features, out_features):
    LHS = {_out: linear(linear_name, [in_features, out_features], [_in])}
    RHS = {
        _0: constant((linear_name, 'weight'), [None, (out_features, in_features)], []),
        _1: constant((linear_name, 'bias'), [None, (out_features,)], []),
        _out: linear_op((linear_name, 'out'), [], [_in, _0, _1])
    }
    return LHS, RHS

@bind_vars
def expand_bn(bn_name, num_features):
    LHS = {_out: bn(bn_name, [num_features], [_in])}
    RHS = {
        _0: constant((bn_name, 'running_mean'), [None, (num_features,)], []),
        _1: constant((bn_name, 'running_var'), [None, (num_features,)], []),
        _2: constant((bn_name, 'weight'), [None, (num_features,)], []),
        _3: constant((bn_name, 'bias'), [None, (num_features,)], []),
        _out: bn_op((bn_name, 'out'), [False], [_in, _0, _1, _2, _3])
    }
    return LHS, RHS
