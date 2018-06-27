from gamma.core import *
from gamma.pytorch import *

################
### Node types
################

prep = node(namedtuple('Prep', ('out_channels')))
classifier = node(namedtuple('Classifier', ('in_channels', 'out_channels')))
residual_block = node(namedtuple('ResidualBlock', ['in_channels', 'out_channels', 'stride', 
                    'h_channels', 'groups', 'act', 'join']))
block = node(namedtuple('Block', ['in_channels', 'out_channels', 'stride', 'h_channels', 'groups', 'act']))
conv_bn = node(namedtuple('ConvBN', ['in_channels', 'out_channels', 'kernel_size', 'stride', 
            'padding', 'groups', 'activation', 'eps']), stride=1, padding=0, groups=1, activation=F.relu, eps=1e-5)
 
######################
### Network defns
######################

def basic_net(blocks, num_classes):
    get_param = lambda t, param_name: t[1]['params'][param_name]
    return pipeline([
        ('prep', prep(get_param(blocks[0], 'in_channels')), ['input']),
        *blocks,
        ('classifier', classifier(get_param(blocks[-1], 'out_channels'), num_classes)),
        ('loss', x_entropy(), ['classifier', 'target'])
    ])

def mobilenetV2(num_classes):
    act = F.relu6
    b = lambda c_in, h, c_out, s: block(c_in, c_out, s, h, h, act)
    r = lambda c_in, h, c_out, s: residual_block(c_in, c_out, s, h, h, act, add()) 
    layers = (
        [b( 16,  16*6,  24, 2)] + [r( 24,  24*6,  24, 1)]*1 +
        [b( 24,  24*6,  32, 2)] + [r( 32,  32*6,  32, 1)]*2 +
        [b( 32,  32*6,  64, 2)] + [r( 64,  64*6,  64, 1)]*3 +
        [b( 64,  64*6,  96, 1)] + [r( 96,  96*6,  96, 1)]*2 +
        [b( 96,  96*6, 160, 2)] + [r(160, 160*6, 160, 1)]*2 +
        [b(160, 160*6, 320, 1)]
    )
    layers = [(f'block_{i}', layer) for (i, layer) in enumerate(layers, 1)]
    net_initial = pipeline([
        ('prep/conv_1', conv_bn(3, 32,  kernel_size=3, padding=1, stride=2, activation=act), ['input']),
        ('prep/conv_2', conv_bn(32, 32, kernel_size=3, padding=1, groups=32, activation=act)),
        ('prep/conv_3', conv_bn(32, 16, kernel_size=1, activation=None)),
        *layers,
        ('classifier/expand', conv_bn(320, 1280, kernel_size=1, activation=act)),
        ('classifier/avg_pool', global_avg_pool()),
        ('classifier/dropout', dropout(p=0.5, inplace=True)),
        ('classifier/fc', linear(1280, num_classes))
    ])   
    rules = [expand_residuals(), replace_identity_shortcuts(), expand_blocks_1_3_1(), expand_conv_bns()]  

    return net_initial, rules, apply_rules(net_initial, rules)

resnet_params = {
    18: [
        [(64, 64, 1)] + [(64, 64, 1)]*1,
        [(64, 128, 2)] + [(128, 128, 1)]*1,
        [(128, 256, 2)] + [(256, 256, 1)]*1,
        [(256, 512, 2)] + [(512, 512, 1)]*1,
    ],
    34: [
        [(64, 64, 1)] + [(64, 64, 1)]*2,
        [(64, 128, 2)] + [(128, 128, 1)]*3,
        [(128, 256, 2)] + [(256, 256, 1)]*5,
        [(256, 512, 2)] + [(512, 512, 1)]*2,
    ],
    50: [
        [(64, 256, 1, 64)] + [(256, 256, 1, 64)]*2,
        [(256, 512, 2, 128)] + [(512, 512, 1, 128)]*3,
        [(512, 1024, 2, 256)] + [(1024, 1024, 1, 256)]*5,
        [(1024, 2048, 2, 512)] + [(2048, 2048, 1, 512)]*2,
    ],
    101: [
        [(64, 256, 1, 64)] + [(256, 256, 1, 64)]*2,
        [(256, 512, 2, 128)] + [(512, 512, 1, 128)]*3,
        [(512, 1024, 2, 256)] + [(1024, 1024, 1, 256)]*22,
        [(1024, 2048, 2, 512)] + [(2048, 2048, 1, 512)]*2,
    ],
    152: [
        [(64, 256, 1, 64)] + [(256, 256, 1, 64)]*2,
        [(256, 512, 2, 128)] + [(512, 512, 1, 128)]*7,
        [(512, 1024, 2, 256)] + [(1024, 1024, 1, 256)]*35,
        [(1024, 2048, 2, 512)] + [(2048, 2048, 1, 512)]*2,
    ]
}

def resnet(depth, num_classes):
    layer_params = resnet_params[depth]
    r = lambda c_in, c_out, s, h=None: residual_block(c_in, c_out, s, h, 1, F.relu, add_relu(True)) 
    layers = [(path(f'layer_{i}', f'block_{j}'), r(*block)) for (i, layer) in enumerate(layer_params, 1) for 
        j, block in enumerate(layer)]
    net_initial = basic_net(layers, num_classes)
    
    rules =  [
        resnet_prep(), 
        resnet_classifier(), 
        expand_residuals(),
        replace_identity_shortcuts(), 
        replace_shortcuts(), 
        expand_blocks_3_3(), 
        expand_blocks_1_3_1(), 
        expand_conv_bns()
    ]

    return net_initial, rules, apply_rules(net_initial, rules)


#######################
### Rewrite rules
#######################

@bind_vars
def resnet_prep(out_channels, _in):
    LHS = {'prep': (prep(out_channels), [_in])}
    RHS = pipeline([
        ('prep', conv_bn(3, 64, (7, 7), stride=2, padding=3, activation=F.relu), [_in]),
        ('maxpool', max_pool((3, 3), stride=2, padding=1)),
    ])
    return LHS, RHS, ('prep', 'maxpool')


@bind_vars
def resnet_classifier(in_channels, out_channels, _in):
    LHS = {'classifier': (classifier(in_channels, out_channels), [_in])}
    RHS = pipeline([
            ('avgpool', global_avg_pool(), [_in]),
            ('fc', linear(in_channels, out_channels)),
    ])
    return LHS, RHS, ('classifier', 'fc')


@bind_vars
def expand_residuals(block_name, in_channels, out_channels, stride, h_channels, groups, act, join, _in):
    LHS = {block_name: (residual_block(in_channels, out_channels, stride, h_channels, groups, act, join), [_in])}
    RHS = pipeline([
        (block_name, block(in_channels, out_channels, stride, h_channels, groups, act), [_in]),
        (path(block_name, 'shortcut'), shortcut(in_channels, out_channels, stride), [_in]),
        (path(block_name, 'join'), join, [block_name, path(block_name, 'shortcut')]),
    ])
    return LHS, RHS, (block_name, path(block_name, 'join'))


@bind_vars
def replace_identity_shortcuts(block_name, in_channels, _in):  
    LHS = {path(block_name, 'shortcut'): (shortcut(in_channels, in_channels, 1), [_in])}
    RHS = {path(block_name, 'shortcut'): (shortcut(in_channels, in_channels, 1, True), [_in])}
    return LHS, RHS

@bind_vars
def replace_shortcuts(block_name, in_channels, out_channels, stride, _in):
    LHS = {path(block_name, 'shortcut'): (shortcut(in_channels, out_channels, stride), [_in])}
    RHS = {path(block_name, 'downsample'): (conv_bn(in_channels, out_channels, (1, 1), stride, activation=None), [_in])}
    return LHS, RHS, (path(block_name, 'shortcut'), path(block_name, 'downsample'))

@bind_vars
def expand_blocks_3_3(block_name, in_channels, out_channels, stride, act, _in):
    LHS = {block_name: (block(in_channels, out_channels, stride, None, 1, act), [_in])}  
    RHS = pipeline([
        (path(block_name, 'conv_1'), conv_bn(in_channels, out_channels, (3, 3), stride=stride, padding=1, activation=act), [_in]),
        (path(block_name, 'conv_2'), conv_bn(out_channels, out_channels, (3, 3), stride=1, padding=1, activation=None))
    ])
    return LHS, RHS, (block_name, path(block_name, 'conv_2'))
                                        
@bind_vars
def expand_blocks_1_3_1(block_name, in_channels, out_channels, stride, h_channels, groups, act, _in):
    LHS = {block_name: (block(in_channels,out_channels, stride, h_channels, groups, act), [_in])}
    RHS = pipeline([
        ('conv_1', conv_bn(in_channels, h_channels, (1, 1), stride=1, activation=act), [_in]),
        ('conv_2', conv_bn(h_channels, h_channels, (3, 3), stride=stride, padding=1, groups=groups, activation=act)),
        ('conv_3', conv_bn(h_channels, out_channels, (1, 1), stride=1, activation=None)) 
    ], prefix=block_name)
    return LHS, RHS, (block_name, path(block_name, 'conv_3'))


@bind_vars
def expand_conv_bns(name, in_channels, out_channels, kernel_size, stride, padding, groups, activation, eps, _in):
    LHS = {name: (conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, activation, eps), [_in])}
    RHS = pipeline([
        ('conv', conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                    bias=False), [_in]),
        ('bn', bn(out_channels, eps=eps)),
        ('act', activation_func(activation_func=activation)),
    ], prefix=name)
    return LHS, RHS, (name, path(name, 'act'))


