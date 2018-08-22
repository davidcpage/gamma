import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
from inspect import signature
from gamma.core import *
from gamma.training import Transducer, Optimizer, transfer, add_, mul_, zeros_like, to_numpy

class transpose(namedtuple('transpose', ('source', 'target'))):
    def __call__(self, data): return data.transpose([self.source.index(x) for x in self.target])

@to_numpy.register(torch.Tensor)
def _(x): 
    return x.detach().cpu().numpy()  


class RecordingContext(object):
    def __init__(self):
        pass 
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass    

class TorchGraph(nn.Module):
    def __init__(self, graph, verbose=False):
        super().__init__()
        self.graph = list(topological_sort(graph))
        self.verbose = verbose
        for n, (a, _) in self.graph: 
            setattr(self, n, a['type'](**a['params']))

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (a, i) in self.graph:
            if self.verbose: print(n)
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

    def params_and_grads(self):
        return ((name, param.data, None if param.grad is None else param.grad.data) for 
                (name, param) in self.named_parameters() if param.requires_grad)

    def param_value(self, node, param_name):
        return to_numpy(getattr(getattr(self, node), param_name))
 
    def set_training(self, mode=True):
        prev_training_state = self.training
        if prev_training_state != mode:
            self.train(mode)
        return prev_training_state

    def recording_context(self):
        return RecordingContext()
 
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

class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

class Permute(nn.Module):
    def __init__(self, before=(), after=()):
        super().__init__()
        self.order = [before.index(x) for x in after if x is not None]
        self.extra = [i for (i,x) in enumerate(after) if x is None]
        
    def forward(self, x):
        x = x.permute(*self.order)
        for i in self.extra:
            x.unsqueeze_(dim=i)
        return x.contiguous()

class SequencewiseBN(nn.BatchNorm1d):
    def forward(self, x):
        *ns, n = x.size()
        x = x.view(-1, n)
        x = super().forward(x)
        return x.view(*ns, n)

class FlattenLast(nn.Module):
    def forward(self, x):
        *sizes, n, m = x.size()
        return x.view(*sizes, n*m)

class NodeDef(namedtuple('NodeDef', ['type', 'params'])):
    def __call__(self, *args, **kwargs): 
        params = self.params.bind(*args, **kwargs)
        params.apply_defaults()
        return {'type': self.type, 'params': dict(params.arguments)}

def node(type_name, arg_names, **defaults):
    return node_def(namedtuple(type_name, arg_names), **defaults)

def node_def(type, **defaults): 
    sig = signature(type)
    if defaults:
        params = [(param.replace(default=defaults[name]) if name in defaults else param) for name, param in sig.parameters.items()]
        sig = sig.replace(parameters=params)
    return NodeDef(type, sig)


identity  = node_def(Identity)  
pool      = node_def(ConcatPool2d)
linear    = node_def(nn.Linear)
bn        = node_def(nn.BatchNorm2d)
conv      = node_def(nn.Conv2d)
conv_op   = node_def(ConvOp)
bn_op     = node_def(BatchNormOp)
linear_op = node_def(LinearOp)
max_pool  = node_def(nn.MaxPool2d)
dropout   = node_def(nn.Dropout)
shortcut  = node_def(Shortcut)
global_avg_pool = node_def(GlobalAvgPool2d)
relu      = node_def(nn.ReLU)

clip = node_def(nn.Hardtanh, inplace=True)
permute = node_def(Permute)
sequencewise_bn = node_def(SequencewiseBN)
flatten_last = node_def(FlattenLast)

relu6     = node_def(nn.ReLU6)
x_entropy = node_def(nn.CrossEntropyLoss)
add       = node_def(Add)
add_relu = node_def(AddRelu)
constant = node_def(Constant)
activation_func = node_def(ActivationFunc)
correct = node_def(Correct)


################
### Training
################

@transfer.register(torch.Tensor)
def _(data, device):
    return data.to(device)

@add_.register(torch.Tensor)
def _(x, a, y):
    if a is 0: return
    if a is 1: x.add_(y)
    else: x.add_(a, y)

@mul_.register(torch.Tensor)
def _(x, y):
    x.mul_(y)

@zeros_like.register(torch.Tensor)
def _(x):
    return torch.zeros_like(x)



class Adam(Optimizer):
    def __init__(self, AdamW = False):
        self.AdamW = AdamW

    def init_state(self, opt_params, model):
        if 'avg_grad' not in opt_params: 
            opt_params['avg_grad'] = zero_param(model)
            opt_params['avg_grad_sq'] = zero_param(model)

    def update(self, p, g, N_step, avg_grad, avg_grad_sq, beta1, beta2, weight_decay, eps, lr):
            scale=256 #loss scaling.. it all cancels out anyhow but g * g seems to produce fp16 underflow.
            step_size = lr * math.sqrt(1 - beta2 ** N_step) / (1 - beta1 ** N_step)
            if weight_decay != 0 and not self.AdamW: g.add_(weight_decay, p)
            g = g*scale                
            avg_grad.mul_(beta1).add_(1 - beta1, g)
            avg_grad_sq.mul_(beta2).addcmul_(1 - beta2, g, g)
            denom = avg_grad_sq.sqrt().add_(eps*scale)
            if self.AdamW: p.add_(-lr*weight_decay, p)
            p.addcdiv_(-step_size, avg_grad, denom)
    