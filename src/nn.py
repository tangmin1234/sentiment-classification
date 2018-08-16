# some code is borrow from https://github.com/allenai/bi-att-flow

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.util import nest
import tensorflow as tf
from general import flatten, reconstruct, add_wd, exp_mask, exp_mask_for_high_rank


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None, kernel_initializer=None):

    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # flat_args[0] : [N*JX*JQ, d]
    if input_keep_prob < 1.0:   
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]
    with tf.variable_scope(scope or 'linear'):
        flat_out = _linear(flat_args, output_size, bias, kernel_initializer=kernel_initializer)
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    if wd:
        add_wd(wd)
    return out 


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def softmax(logits, mask=None, scope=None, rescale=False, dim=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        if rescale:
            assert dim is not None
            logits = tf.divide(logits, tf.ones_like(logits, dtype=tf.float32) * tf.sqrt(dim))
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)
        return out

def softsel(target, logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out

def double_proj_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "double_proj_logits"):
        first = linear(args[0], size, False, bias_start, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        second = linear(args[1], size, False, bias_start, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        logits = tf.matmul(first, second, transpose_b=True) # [N, JX, JQ]
        return logits

def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                               wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, 1, False, bias_start=bias_start, squeeze=True, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            second = exp_mask(second, mask)
        return second

def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits

def bilinear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or 'bilinear'):
        proj = linear([args[0]], size, False, bias_start=bias_start, scope="proj", 
                    wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        args[1] = dropout(args[1], input_keep_prob, is_train)
        logits = tf.matmul(proj, args[1], transpose_b=True)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits

def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank-1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "sum"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, 
                                input_keep_prob=input_keep_prob, is_train=is_train)
    elif func == 'double':
        d = args[0].get_shape()[-1]
        return double_linear_logits(args, d, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, 
                                    input_keep_prob=input_keep_prob, is_train=is_train)
    elif func == 'tri_linear':
        if len(args) == 2:
            new_arg = args[0] * args[1]
        else:
            new_arg = args[2]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, 
                                input_keep_prob=input_keep_prob, is_train=is_train)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, 
                                input_keep_prob=input_keep_prob, is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        logits = tf.matmul(args[0], args[1], transpose_b=True)
        return logits

    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, 
                                input_keep_prob=input_keep_prob, is_train=is_train)
    elif func == "double_proj":
        return double_proj_logits(args, size, bias, bias_start=bias_start, scope=scope, 
                        mask=mask, wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        logits = bilinear_logits(args, size, bias, bias_start=bias_start, scope=scope, 
                                    mask=mask, wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        return logits
    else:
        raise ValueError("Invalid logit function type...")


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur

def highway_network2(args, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network2"):
        d = args[0].get_shape().as_list()[-1]
        trans = linear(args[0], d, bias, bias_start=bias_start, 
                        scope=scope or 'trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        trans = tf.nn.sigmoid(trans)
        out = (1 - trans) * args[0] + trans * args[1]
        return out

def highway_network3(args, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network2"):
        d = args[0].get_shape().as_list()[-1]
        trans = linear(args, d, bias, bias_start=bias_start, 
                        scope=scope or 'trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        trans = tf.nn.sigmoid(trans)
        out = (1 - trans) * args[0] + trans * args[1]
        return out

def gather(params, indices):
    d = params.get_shape().as_list()[-1]
    N = tf.shape(indices)[0]
    k = indices.get_shape().as_list()[-1]
    batch_indices = tf.expand_dims(tf.tile(tf.range(N), [k]), 1) # [N*k, 1]
    indices = tf.concat([batch_indices, tf.reshape(indices, [N*k, 1])], 1)
    out = tf.gather_nd(params, indices)
    out = tf.reshape(out, [N, k, d])
    return out

def pool(in_, window_type, pooling_type="max", padding="SAME", strides=None, name=None):
    with tf.name_scope(name or "pool"):
        out = tf.nn.pool(in_, window_shape, pooling_type, padding)
        return out


def conv1d(in_, filter_size, height, padding, wd=0.0, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
        if wd:
            add_wd(wd)
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, wd=0.0, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, wd=wd, 
                            is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height))
            outs.append(out)
        concat_out = tf.concat(outs, 2)
        return concat_out

def conv2d_v2(in_, filter_size, height, padding, dilation_rate=None, wd=0.0, is_train=None, keep_prob=1.0, scope=None, nonlinear=True):
    with tf.variable_scope(scope or "conv2d"):
        N = tf.shape(in_)[0]
        in_ = tf.expand_dims(in_, -1)
        num_channels = in_.get_shape().as_list()[-1]
        filter_ = tf.get_variable("filter_", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1]
        dilation_rate = [1, dilation_rate or 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        out = tf.nn.convolution(in_, filter_, padding, strides=strides, dilation_rate=dilation_rate) + bias
        if nonlinear:
            out = tf.nn.relu(out)
        out = tf.reshape(out, [N, -1, filter_size])
        if wd:
            add_wd(wd)
        return out


def conv2d(in_, filter_size, height, padding, dilation_rate=None, wd=0.0, is_train=None, keep_prob=1.0, scope=None, nonlinear=True):
    with tf.variable_scope(scope or "conv2d"):
        num_channels = in_.get_shape().as_list()[-1]
        N = tf.shape(in_)[0]
        in_ = tf.expand_dims(in_, 1)
        filter_ = tf.get_variable("filter_", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1]
        dilation_rate = [1, dilation_rate or 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        out = tf.nn.convolution(in_, filter_, padding, strides=strides, dilation_rate=dilation_rate) + bias
        if nonlinear:
            out = tf.nn.tanh(out)
        out = tf.reshape(out, [N, -1, filter_size])
        if wd:
            add_wd(wd)
        return out

def multi_conv2d(in_, filter_sizes, heights, padding, dilation_rates=None, is_train=None, keep_prob=1.0, wd=0.0, scope=None, nonlinear=True, mask=None):
    with tf.variable_scope(scope or "multi_conv2d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_sizes == 0:
                continue
            dilation_rate = dilation_rates[i] if dilation_rates is not None else None
            out = conv2d(in_, filter_size, height, padding, dilation_rate=dilation_rate, wd=wd, 
                        is_train=is_train, keep_prob=keep_prob, scope="conv2d_{}".format(height), nonlinear=nonlinear)
            outs.append(out)
        concat_out = tf.concat(outs, 2)
        if mask is not None:
            concat_out = concat_out * tf.expand_dims(tf.to_float(mask), -1)
        return concat_out

def get_multi_channel_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    """
    Args:
        h: sequence embedding, shape = [N, J1, J2, d]
        u: sequence embedding, shape = [N, J2, J2, d]

    """
    N, J1, J2 = tf.shape(args[0])[0], tf.shape(args[0])[1], tf.shape(args[0])[2]
    d = args[0].get_shape().as_list()[-1]

    with tf.variable_scope(scope or "Multi_channel_logits"):
        first = tf.tanh(linear(args, d, True, bias_start=bias_start, scope='first', wd=wd, 
                                input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, d, True, bias_start=bias_start, scope='second', wd=wd, 
                        input_keep_prob=input_keep_prob, is_train=is_train)
        
        if mask is not None:
            second = exp_mask_for_high_rank(second, mask)
        return second