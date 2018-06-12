import functools
import time
from collections import namedtuple
import torch
import numpy as np
from .pytorch import to_numpy


################
# Transducers
################
def compose(*funcs):
    f, *fs = tuple(reversed(funcs))
    return lambda *args, **kwargs: functools.reduce(lambda acc, f: f(acc), fs, f(*args, **kwargs))


def reduce(reducer, iterable, init=None):
    acc = reducer.initialize(init)
    for item in iterable:
        acc, reduced = reducer.step(acc, item)
        if reduced:
            break
    return reducer.finalize(acc)


class Transducer:
    def __init__(self, reducer):
        self.reducer = reducer

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

##################
# Training
##################   

class Forward(Reducer):
    def __init__(self, training):
        self.training = training
    
    def initialize(self, state):
        state['model'].train(self.training)
        return state
    
    @staticmethod
    def step(state, inputs):
        output = state['model']({'input': inputs[0].half().cuda(), 'target': inputs[1].cuda()})
        return state, False


def is_correct(outputs): return to_numpy(outputs['classifier']).argmax(
    axis=1) == to_numpy(outputs['target'])

class LogStats(Transducer):
    def __init__(self, examples_per_epoch):
        self.examples_per_epoch = examples_per_epoch

    def initialize(self, state):
        if 'stats' not in state: state['stats'] = []
        state['stats'].append(
            {'progress': 0, 'total': 0, 'correct': 0, 'total_loss': 0, 'start_time': time.time()}
        )
        return self.reducer.initialize(state)

    def step(self, state, inputs):
        state, reduced = self.reducer.step(state, inputs)
        outputs = state['model'].cache
        n = outputs['target'].shape[0]
        stats = state['stats'][-1]
        stats['total_loss'] += to_numpy(outputs['loss'])*n
        stats['total'] += n
        stats['correct'] += is_correct(outputs).sum()
        stats['progress'] = stats['total']/self.examples_per_epoch
        return state, reduced

    def finalize(self, state):
        stats = state['stats'][-1]
        stats['end_time'] = time.time()
        stats.update({
            'time': stats['end_time']-stats['start_time'],
            'acc': stats['correct']/stats['total'],
            'loss': stats['total_loss']/stats['total']
        })
        return self.reducer.finalize(state)


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
        opt = state['optimizer']
        v, momentum, lr, weight_decay = opt['v'], opt['momentum'], opt['lr'], opt.get(
            'weight_decay', 0)
        for name, param in state['model'].named_parameters():
            g = param.grad.data
            mom = v[name]
            if weight_decay != 0:
                g.add_(weight_decay, param.data)
            mom.mul_(momentum).add_(g)
            g = g.add(momentum, mom)
            param.data.add_(-lr, g)
        return state, reduced

class piecewise_linear(namedtuple('piecewise_linear', ('knots', 'vals'))):
    def __call__(self, x): return np.interp([x], self.knots, self.vals)[0]


def plot_lr_schedule(lr_schedule, epochs, ax):
    return ax.plot(*zip(*[(x, lr_schedule(x)) for x in np.arange(0, epochs, 0.1)]))

class LRScheduler(Transducer):
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule
                               
    def step(self, state, inputs):
        progress = state['epoch'] + state['stats'][-1]['progress']
        state['optimizer']['lr'] = self.lr_schedule(progress)
        return self.reducer.step(state, inputs)


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

class EpochRunner(Reducer):
    @staticmethod
    def print_format(*vals):
        #pylint: disable=E1101
        formats = {str: '{:10}', int: '{:10d}',
                   np.float64: '{:10.4f}', float: '{:10.4f}'}
        #pylint: enable=E1101
        print(' '.join(formats[type(v)].format(v) for v in vals))

    def __init__(self, train_step, test_step):
        self.train_step = train_step
        self.test_step = test_step

    def initialize(self, state):
        print(f'Starting training at '+time.strftime('%Y-%m-%d %H:%M:%S'))
        self.print_format('epoch', 'lr', 'trn_time', 'trn_loss',
                          'trn_acc', 'val_time', 'val_loss', 'val_acc')
        return state

    def step(self, state, item):
        train, test = item
        state = reduce(self.train_step, train, state)
        state = reduce(self.test_step, test, state)
        state['epoch'] += 1
        stats = [state['epoch'], state['optimizer']['lr']] + [state['stats'][i][k]
                                                              for i in [-2, -1] for k in ['time', 'loss', 'acc']]
        self.print_format(*stats)

        return state, False

    def finalize(self, state):
        print(f'Finished training at '+time.strftime('%Y-%m-%d %H:%M:%S'))
        return state
