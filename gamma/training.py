import functools
import time
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


class Nesterov(Transducer):
    def initialize(self, state):
        state = self.reducer.initialize(state)
        opt_params = state['optimizer']
        if 'v' not in opt_params:
            #pylint: disable=E1101
            opt_params['v'] = {k: torch.zeros_like(
                v) for k, v in state['model'].named_parameters() if v.requires_grad}
            #pylint: enable=E1101
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

    
