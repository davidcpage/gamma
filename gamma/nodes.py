from collections import namedtuple
from inspect import signature

class NodeDef(namedtuple('NodeDef', ['type', 'params'])):
    def __call__(self, *args, **kwargs): 
     #   print(self.type)
        params = self.params.bind(*args, **kwargs)
        params.apply_defaults()
        return {'type': self.type, 'params': dict(params.arguments)}

def node(type_name, arg_names=(), **defaults):
    arg_names = list(arg_names) + [k for k in defaults.keys() if k not in arg_names]
    return node_def(namedtuple(type_name, arg_names), **defaults)

def node_def(type, **defaults): 
    sig = signature(type)
    if defaults:
        params = [(param.replace(default=defaults[name]) if name in defaults else param) for name, param in sig.parameters.items()]
        sig = sig.replace(parameters=params)
    return NodeDef(type, sig)

add = node('Add', inplace=True)
add_relu = node('AddRelu', inplace=False)
bn = node('BatchNorm2d', ['num_features'], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
clip = node('Clip', min_val=-1, max_val=1, inplace=False, min_value=None, max_value=None)
concat_pool_2d = node('ConcatPool2d')
conv = node('Conv2d', ['in_channels', 'out_channels', 'kernel_size'],  stride=1, padding=0, dilation=1, groups=1, bias=True)
correct = node('Correct')
dropout = node('Dropout', p=0.5, inplace=False)
global_avg_pool = node('GlobalAvgPool2d')

identity = node('Identity')
linear = node('Linear', ['in_features', 'out_features'], bias=True)
max_pool = node('MaxPool2d', ['kernel_size'], stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
relu = node('ReLU', inplace=False)
relu6 = node('ReLU6', inplace=False)
shortcut = node('Shortcut', ['in_channels', 'out_channels', 'stride'], identity=False)
x_entropy = node('CrossEntropyLoss', weight=None, size_average=True, ignore_index=-100, reduce=True)

