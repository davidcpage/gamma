import functools
import math
from collections import namedtuple
import numpy as np


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

@functools.singledispatch
def transfer(data, device):
    raise NotImplementedError

@functools.singledispatch
def add_(x, a, y):
    #x += a*y
    raise NotImplementedError

@functools.singledispatch
def mul_(x, y):
    #x *= y
    raise NotImplementedError

@functools.singledispatch
def zeros_like(x):
    raise NotImplementedError


@functools.singledispatch
def to_numpy(x):
    raise NotImplementedError


def zero_param(model):
    return {k: zeros_like(v) for k, v, _ in model.params_and_grads()}


##################
# Training
##################   

class Forward(Reducer):
    def __init__(self, training):
        self.training = training
    
    def initialize(self, state):
        state['processed'] = 0
        state['batches'] = 0
        self.prev_training_mode = state['model'].set_training(self.training)
        return state
    
    def step(self, state, inputs):
        state['processed'] += len(inputs[0])
        state['batches'] += 1
        device = state['device']
        output = state['model']({'input': transfer(inputs[0], device), 'target': transfer(inputs[1], device)})
        return state, False

    def finalize(self, state):
        state['model'].set_training(self.prev_training_mode)
        return state

class Memo(Transducer):
  def __init__(self, paths):
    self.paths = paths
    
  def initialize(self, state):
    if 'memo' not in state: state['memo'] = {path: [] for path in self.paths}
    return self.reducer.initialize(state)
  
  def step(self, state, inputs):
    state, reduced = self.reducer.step(state, inputs)
    for path in self.paths:
      state['memo'][path].append(state['model'].param_value(*path))
    return state, reduced
  

class Backward(Transducer):
    def initialize(self, state):
        state = self.reducer.initialize(state)
        return state

    def step(self, state, inputs):
        state['model'].zero_grad()
        with state['model'].recording_context():
            state, reduced = self.reducer.step(state, inputs)
        state['model'].cache['loss'].backward()
        return state, reduced

    def finalize(self, state):
        return self.reducer.finalize(state)
        

class Optimizer(Transducer):
    def init_state(self, opt_params, model):
        pass

    def update(self, param, grad, **kwargs):
        pass

    def initialize(self, state):
        state = self.reducer.initialize(state)
        if 'optimizer' not in state: state['optimizer'] = {}
        if 'N_step' not in state['optimizer']: state['optimizer']['N_step'] = 0
        self.init_state(state['optimizer'], state['model'])
        return state

    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        if reduced: return state, reduced
        state['optimizer']['N_step'] += 1
        for name, param, grad in state['model'].params_and_grads():
            opt_params = {k: v[name] if isinstance(v, dict) else v for (k, v) in state['optimizer'].items()}
            self.update(param, grad, **opt_params)
        return state, reduced


class Nesterov(Optimizer):
    def init_state(self, opt_params, model):
        if 'v' not in opt_params: opt_params['v'] = zero_param(model)

    def update(self, p, g, v, momentum, lr, weight_decay, **kwargs):
        add_(g, weight_decay, p)
        mul_(v, momentum)
        add_(v, 1, g)
        add_(g, momentum, v)
        add_(p, -lr, g)


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

    
