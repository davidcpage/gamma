import functools
import math
from collections import namedtuple
import torch
import numpy as np
from .pytorch import to_numpy


################
# Transducers
################

class compose(namedtuple('compose', ('fs'))):
    def __new__(cls, *args): 
        return super().__new__(cls, args)

    def __call__(self, *args, **kwargs):
        f, *fs = tuple(reversed(self.fs))  
        return functools.reduce(lambda acc, f: f(acc), fs, f(*args, **kwargs))

def reduce(reducer, iterable, init=None):
    acc = reducer.initialize(init)
    for item in iterable:
        acc, reduced = reducer.step(acc, item)
        if reduced:
            break
    return reducer.finalize(acc)


class Transducer:
    def initialize(self, state):
        return self.reducer.initialize(state)

    def step(self, state, item):
        return self.reducer.step(state, item)

    def finalize(self, state):
        return self.reducer.finalize(state)

    def __call__(self, reducer):
        self.reducer = reducer
        return self

#    def __repr__(self):
 #       arg_string = ', '.join(f'{name!s}={value!r}' for name, value in self.params())
 #       return f'{type(self).__name__}({arg_string})'

class Reducer:
    @staticmethod
    def initialize(state):
        return state

    def step(self, state, item):
        raise NotImplementedError

    @staticmethod
    def finalize(state):
        return state

##################
# Training
##################   

class Forward(Reducer):
    def __init__(self, training):
        self.training = training
    
    def initialize(self, state):
        state['model'].train(self.training)
        state['processed'] = 0
        state['batches'] = 0
        return state
    
    @staticmethod
    def step(state, inputs):
        state['processed'] += len(inputs[0])
        state['batches'] += 1
        device = state['device']
        output = state['model']({'input': inputs[0].to(device), 'target': inputs[1].to(device)})
        return state, False


class Memo(Transducer):
  def __init__(self, paths):
    self.paths = paths
    
  def initialize(self, state):
    if 'memo' not in state: state['memo'] = {path: [] for path in self.paths}
    return self.reducer.initialize(state)
  
  def step(self, state, inputs):
    state, reduced = self.reducer.step(state, inputs)
    for (k1, k2) in self.paths:
      state['memo'][(k1, k2)].append(to_numpy(getattr(getattr(state['model'], k1), k2)))
    return state, reduced
  

class Backward(Transducer):
    def initialize(self, state):
        state = self.reducer.initialize(state)
        state['model'].train(True)
        return state

    def step(self, state, inputs):
        state['model'].zero_grad()
        state, reduced = self.reducer.step(state, inputs)
        state['model'].cache['loss'].backward()
        return state, reduced

def zero_param(model):
    #pylint: disable=E1101
    return {k: torch.zeros_like(v) for k,v in model.named_parameters() if v.requires_grad}
    #pylint: enable=E1101

class Nesterov(Transducer):
    def initialize(self, state):
        state = self.reducer.initialize(state)
        params = state['optimizer']
        if 'v' not in params: params['v'] = zero_param(state['model'])
        return state

    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        if reduced: return state, reduced
        opt_params = state['optimizer']
        v, momentum, lr, weight_decay = [opt_params[k] for k in ['v', 'momentum', 'lr', 'weight_decay']]
        for name, param in state['model'].named_parameters():
            p, g = param.data, param.grad.data
            mom = v[name]
            if weight_decay != 0: g.add_(weight_decay, p)
            mom.mul_(momentum).add_(g)
            g = g.add(momentum, mom)
            p.add_(-lr, g)
        return state, reduced

class Adam(Transducer):
    def __init__(self, AdamW = False):
        self.AdamW = AdamW

    def initialize(self, state):
        state = self.reducer.initialize(state)
        params = state['optimizer']
        if 'N_step' not in params: 
            params['N_step'] = 0
            params['avg_grad'] = zero_param(state['model'])
            params['avg_grad_sq'] = zero_param(state['model'])
        return state

    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        if reduced: return state, reduced
        opt_params = state['optimizer']
        opt_params['N_step'] += 1
        N_step, avg_grad, avg_grad_sq, beta1, beta2, weight_decay, eps, lr = [opt_params[k] for k in ['N_step', 
            'avg_grad', 'avg_grad_sq', 'beta1', 'beta2', 'weight_decay', 'eps', 'lr']]
        step_size = lr * math.sqrt(1 - beta2 ** N_step) / (1 - beta1 ** N_step)     
        scale=256 #loss scaling.. it all cancels out anyhow but g * g seems to produce fp16 underflow.
        eps = eps*scale
        for k, param in state['model'].named_parameters():
            p, g = param.data, param.grad.data
            if weight_decay != 0 and not self.AdamW: g.add_(weight_decay, p)
            g = g*scale                
            avg_grad[k].mul_(beta1).add_(1 - beta1, g)
            avg_grad_sq[k].mul_(beta2).addcmul_(1 - beta2, g, g)
            denom = avg_grad_sq[k].sqrt().add_(eps)
            if self.AdamW: p.add_(-lr*weight_decay, p)
            p.addcdiv_(-step_size, avg_grad[k], denom)
        return state, reduced

class piecewise_linear(namedtuple('piecewise_linear', ('knots', 'vals'))):
    def __call__(self, t): 
        return np.interp([t], self.knots, self.vals)[0]
 
def plot_lr_schedule(lr_schedule, epochs, ax):
    return ax.plot(*zip(*[(x, lr_schedule(x)) for x in np.arange(0, epochs, 0.1)]))


class LRScheduler(Transducer):
    def __init__(self, params):
        self.params = params

    def initialize(self, state):
        if 'optimizer' not in state: state['optimizer'] = {}
        return self.reducer.initialize(state)

    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        progress = state['epoch'] + state['batches'] / state['epoch_length']
        for k, f in self.params.items():
            state['optimizer'][k] = f(progress) if callable(f) else f
        return state, reduced


class EarlyStop(Transducer):
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def initialize(self, state):
        self.counter = 0
        return self.reducer.initialize(state)

    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        self.counter += 1
        reduced = reduced or (self.counter == self.num_batches)
        return state, reduced

    
