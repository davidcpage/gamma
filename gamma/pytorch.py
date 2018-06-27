import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
from inspect import signature
from gamma.core import *

class TorchGraph(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = dict(topological_sort(graph))
        for n, (a, _) in self.graph.items(): 
            setattr(self, n, a['type'](**a['params']))

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (a, i) in self.graph.items():
            #print(n)
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

def rename(state_dict, rules):
    import parse
    rules = [(parse.compile(LHS), RHS) for (LHS, RHS) in rules]
    parses = ((LHS.parse(k), RHS, v) for k, v in state_dict.items() for LHS, RHS in rules)
    return {RHS.format(*p.fixed, **p.named): v for p, RHS, v in parses if p}


def load_state(net, state_dict, sep='/'):
    for key, val in state_dict.items():
        *head, tail = key.split(sep)
        mod = getattr(net, sep.join(head))
        #'https://download.pytorch.org/models/resnet152-b121ed2d.pth' stores parameters as tensors...
        if isinstance(getattr(mod, tail), nn.Parameter) and not isinstance(val, nn.Parameter):
            val = nn.Parameter(val) 
        setattr(mod, tail, val)    


def to_numpy(x): return x.detach().cpu().numpy()    

class Identity(nn.Module):
    def forward(self, x): return x

class ActivationFunc(nn.Module):
    def __init__(self, activation_func=None, inplace=True):
        super().__init__()
        self.activation_func = activation_func
        self.inplace=inplace
    def forward(self, x):
        if self.activation_func is None: return x
        return self.activation_func(x, inplace=self.inplace)


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, identity=False):
        super().__init__()
        self.in_channels, self.out_channels, self.stride, self.identity = in_channels, out_channels, stride, identity
    def forward(self, x):
        if self.identity: return x
        raise NotImplementedError
        

class Add(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x, y): 
        if self.inplace: return x.add_(y)
        else: return x + y


class AddRelu(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace=inplace
    def forward(self, x, y): 
        if self.inplace: 
            return F.relu_(x.add_(y))
        return F.relu(x.add(y))
    
    
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


class NodeDef(namedtuple('NodeDef', ['type', 'params'])):
    def __call__(self, *args, **kwargs): 
        params = self.params.bind(*args, **kwargs)
        params.apply_defaults()
        return {'type': self.type, 'params': dict(params.arguments)}


def node(type, **defaults): 
    sig = signature(type)
    if defaults:
        params = [(param.replace(default=defaults[name]) if name in defaults else param) for name, param in sig.parameters.items()]
        sig = sig.replace(parameters=params)
    return NodeDef(type, sig)

identity  = node(Identity)  
pool      = node(ConcatPool2d)
linear    = node(nn.Linear)
bn        = node(nn.BatchNorm2d)
conv      = node(nn.Conv2d)
conv_op   = node(ConvOp)
bn_op     = node(BatchNormOp)
linear_op = node(LinearOp)
max_pool  = node(nn.MaxPool2d)
dropout   = node(nn.Dropout)
shortcut  = node(Shortcut)
global_avg_pool = node(GlobalAvgPool2d)
relu      = node(nn.ReLU)
relu6     = node(nn.ReLU6)
x_entropy = node(nn.CrossEntropyLoss)
add       = node(Add)
add_relu = node(AddRelu)
constant = node(Constant)
activation_func = node(ActivationFunc)

