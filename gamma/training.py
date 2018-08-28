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
        self.prev_training_mode = state['model'].set_training(self.training)
        return state
    
    def step(self, state, inputs):
        state['output'] = state['model'](inputs)
        return state, False

    def finalize(self, state):
        state['model'].set_training(self.prev_training_mode)
        return state


class Memo(Transducer):
    # funcs is a dictionary of {key: func}
    # Memo records the value of func(state) in state['memo'][key]
    def __init__(self, funcs):
        self.funcs = funcs
    
    def initialize(self, state):
        if 'memo' not in state: 
            state['memo'] = {k: [] for k in self.funcs.keys()}
        return self.reducer.initialize(state)
  
    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        for k, f in self.funcs.items():
            state['memo'][k].append(f(state))
        return state, reduced

class Backward(Transducer):
    def __init__(self, loss_node='loss', loss_scale=None):
        self.loss_node = loss_node
        self.loss_scale = loss_scale

    def initialize(self, state):
        state = self.reducer.initialize(state)
        return state

    def step(self, state, inputs):
        state['model'].zero_grad()
        with state['model'].recording_context():
            state, reduced = self.reducer.step(state, inputs)
            loss = state['output'][self.loss_node]
            if self.loss_scale is not None: loss = loss*self.loss_scale
        loss.backward()
        return state, reduced

    def finalize(self, state):
        return self.reducer.finalize(state)

   
def progress(state, inputs):
    return state['epoch'] + inputs['batch_idx']/inputs['total_batches']

class Optimizer(Transducer):
    def __init__(self, **params):
        self.params = params

    def init_state(self, opt_state, model):
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
        t = progress(state, inputs)
        opt_params = state['optimizer']
        for k, f in self.params.items():
            opt_params[k] = f(t) if callable(f) else f
        opt_params['N_step'] += 1
        for name, param, grad in state['model'].params_and_grads():
            self.update(param, grad, **{k: v[name] if isinstance(v, dict) else v for (k, v) in opt_params.items()})
        return state, reduced


class Nesterov(Optimizer):
    def init_state(self, opt_state, model):
        if 'v' not in opt_state: opt_state['v'] = zero_param(model)

    def update(self, p, g, v, momentum, lr, weight_decay, **kwargs):
        add_(g, weight_decay, p)
        mul_(v, momentum)
        add_(v, 1, g)
        add_(g, momentum, v)
        add_(p, -lr, g)


class piecewise_linear(namedtuple('piecewise_linear', ('knots', 'vals'))):
    def __call__(self, t): 
        return np.interp([t], self.knots, self.vals)[0]
    def __mul__(self, r):
        return piecewise_linear(self.knots, [x*r for x in self.vals])
    __rmul__ = __mul__

def plot_lr_schedule(lr_schedule, epochs, ax):
    return ax.plot(*zip(*[(x, lr_schedule(x)) for x in np.arange(0, epochs, 0.1)]))


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

class transpose(namedtuple('transpose', ('source', 'target'))):
    def __call__(self, data): return data.transpose([self.source.index(x) for x in self.target])
