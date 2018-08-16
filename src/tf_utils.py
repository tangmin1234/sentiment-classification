import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers import CudnnGRU, CudnnLSTM
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from nn import get_logits, get_multi_channel_logits, softsel, dropout, linear
from general import exp_mask, flatten, reconstruct, add_wd
import tensorflow.contrib as tc
from tensorflow.python.ops import rnn_cell_impl



class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0, 
             seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob, 
                                                       state_keep_prob=state_keep_prob, seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                       for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state

def cudnn_rnn(rnn_type, inputs, length, hidden_size, num_layers=1, 
        dropout_keep_prob=1.0, concat=True, initial_state=None, 
        kernel_initializer=tf.random_normal_initializer(stddev=0.1), wd=0.0, is_train=False, scope=None):
    with tf.variable_scope(scope or 'cudnn_rnn'):
        direction = "bidirectional" if 'bi' in rnn_type else "unidirectional"
        input_size = inputs.get_shape().as_list()[-1]
        if rnn_type.endswith('gru'):
            rnn = CudnnGRU(num_layers=num_layers, num_units=hidden_size, 
                            input_mode='linear_input', direction=direction, 
                            dropout=1-dropout_keep_prob, name='rnn')
        
        elif rnn_type.endswith('lstm'):
            rnn = CudnnLSTM(num_layers=num_layers, num_units=hidden_size, 
                            input_mode='linear_input', direction=direction, 
                            dropout=1-dropout_keep_prob, name='rnn')
        else:
            raise NotImplementedError("{} is not supported.".format(rnn_type))
        inputs = dropout(inputs, dropout_keep_prob, is_train)
        outputs, _ = rnn(tf.transpose(inputs, [1, 0, 2]))
        outputs = tf.transpose(outputs, [1, 0, 2]) # [N, JX, 2*d]
        output_h = None
        if wd:
            add_wd(wd)
        return outputs, output_h


def gather(params, indices):
    d = params.get_shape().as_list()[-1]
    N = tf.shape(indices)[0]
    indices = tf.expand_dims(indices, 1) # [N, 1]
    batch_indices = tf.expand_dims(tf.range(N), 1) # [N, 1]
    indices = tf.concat([batch_indices, indices], 1) # [N, 2]
    out = tf.gather_nd(params, indices)
    out = tf.reshape(out, [N, d])
    return out


def rnn(rnn_type, inputs, length, hidden_size, num_layers=1, state_keep_prob=1.0, 
        dropout_keep_prob=None, concat=True, initial_state=None, 
        kernel_initializer=tf.random_normal_initializer(stddev=0.1), wd=0.0, is_train=False, scope=None):
    with tf.variable_scope(scope or 'rnn'):
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, num_layers, dropout_keep_prob, kernel_initializer=kernel_initializer, is_train=is_train)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32, initial_state=initial_state)
            if rnn_type.endswith('lstm'):
                h = states[0][1]
                # h = [state.h for state in states]
                state = h
        else:
            cell_fw = get_cell(rnn_type, hidden_size, num_layers, dropout_keep_prob, state_keep_prob, kernel_initializer=kernel_initializer, is_train=is_train)
            cell_bw = get_cell(rnn_type, hidden_size, num_layers, dropout_keep_prob, state_keep_prob, kernel_initializer=kernel_initializer, is_train=is_train)
            if initial_state is not None:
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32, 
                    initial_state_fw=initial_state[:, :hidden_size], initial_state_bw=initial_state[:, hidden_size:]
                )
            else:
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
                )
            state_fw, state_bw = states
            if rnn_type.endswith('lstm'):
                h_fw = state_fw[0][1]
                h_bw = state_bw[0][1]
                # h_fw = [state_fw.h for state_fw in states_fw]
                # h_bw = [state_bw.h for state_bw in states_bw]
                state_fw, state_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                state = tf.concat([state_fw, state_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                state = state_fw + state_bw
            if wd:
                add_wd(wd)
        return outputs, state


def get_cell(rnn_type, hidden_size, num_layers=1, dropout_keep_prob=None, state_keep_prob=None, kernel_initializer=None, is_train=False):
    cells = []
    if rnn_type.endswith('lstm'):
        state_is_tuple = True
    else:
        state_is_tuple = False
    for i in range(num_layers):
        if rnn_type.endswith('wlstm'):
            cell = MultiWeightCell(num_units=hidden_size)
        elif rnn_type.endswith('lstm'):
            cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=state_is_tuple)
        elif rnn_type.endswith('gru'):
            cell = tc.rnn.GRUCell(num_units=hidden_size)
        elif rnn_type.endswith('rnn'):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        elif rnn_type.endswith('attgru'):
            cell = AttentionCell(num_units=hidden_size)
        elif rnn_type.endswith('sru'):
            cell = SRUCell(num_units=hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=dropout_keep_prob, state_keep_prob=state_keep_prob)
        cells.append(cell)
    cell = tc.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
    return cell

def l2_normalize(in_, proj=True, input_keep_prob=1.0, wd=0.0, is_train=None, scope=None):
    d = in_.get_shape().as_list()[-1]
    with tf.variable_scope(scope or "l2_normalize"):
        if proj:
            in_ = F(in_, d, scope='in_', input_keep_prob=input_keep_prob, wd=wd, is_train=is_train, use_bias=True)
        in_2 = flatten(in_, 1)
        out = tf.nn.l2_normalize(in_2, 1)
        out = reconstruct(out, in_, 1)
        return out

def create_calc_similarity_fn(cfg, is_train=None):
    def calc_similarity_fn(h, u, logit_type=None, scope=None):
        d_h = u.get_shape().as_list()[-1]
        if logit_type == 'double' or logit_type == 'tri_linear':
            J1 = tf.shape(h)[1]
            J2 = tf.shape(u)[1]
            aug_h = tf.tile(tf.expand_dims(h, 2), [1, 1, J2, 1])
            aug_u = tf.tile(tf.expand_dims(u, 1), [1, J1, 1, 1])
        if logit_type == 'dot':
            out = get_logits([h, u], None, True, scope=scope, wd=cfg.wd, 
                                input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='dot') # [N, JX, JQ]
        elif logit_type == 'double':
            out = get_logits([aug_h, aug_u], None, True, scope=scope, wd=cfg.wd, 
                                input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='double')
        elif logit_type == 'tri_linear':
            out = get_logits([aug_h, aug_u, aug_h * aug_u], None, True, scope=scope, wd=cfg.wd, 
                                input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='tri_linear')
        elif logit_type == 'projected_bilinear':
            with tf.variable_scope("projection"):
                h = F(h, d_h, scope='h', input_keep_prob=cfg.input_keep_prob, wd=cfg.wd, is_train=is_train, use_bias=True)
                tf.get_variable_scope().reuse_variables()
                u = F(u, d_h, scope='h', input_keep_prob=cfg.input_keep_prob, is_train=is_train, use_bias=True)
            out = get_logits([h, u], d_h, True, scope=scope, wd=cfg.wd, 
                            input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='proj') # [N, JX, JQ] 
        elif logit_type == 'bilinear':
            out = get_logits([h, u], d_h, True, scope=scope, wd=cfg.wd, 
                            input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='proj') # [N, JX, JQ] 
        elif logit_type == 'double_proj':
            out = get_logits([h, u], d_h, True, scope=scope, wd=cfg.wd, 
                            input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='double_proj')
        elif logit_type == 'cosine':
            with tf.variable_scope("projection"):
                h = l2_normalize(h, proj=False, input_keep_prob=cfg.input_keep_prob, is_train=is_train, wd=cfg.wd, scope=scope + 'normal_h')
                tf.get_variable_scope().reuse_variables()
                u = l2_normalize(u, proj=False, input_keep_prob=cfg.input_keep_prob, is_train=is_train, scope=scope + 'normal_h')
            out = get_logits([h, u], None, True, scope=scope, wd=cfg.wd, 
                            input_keep_prob=cfg.input_keep_prob, is_train=is_train, func='dot') # [N, JX, JQ]
        else:
            raise NotImplementedError("not implemented!")
        return out

    return calc_similarity_fn

def create_compose_fn(cfg):
    def compose_fn(h, u_a, compose_type='elemwise-multiplication', mask=None, scope=None):
        with tf.variable_scope(scope or "compose"):
            if compose_type == "elemwise-multiplication":
                out = h * u_a
            elif compose_type == "elemwise-addition":
                out = h + u_a
            elif compose_type == "concatenate":
                out = tf.concat([h, u_a], -1)
            elif compose_type == "tri_concatenate":
                out = tf.concat([h, u_a, h * u_a], -1)
            else:
                raise NotImplementedError("not implemented!")  
            return out
    return compose_fn

def get_optimizer(cfg, learning_rate):
    if cfg.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif cfg.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    elif cfg.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif cfg.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    return optimizer

def create_calc_multi_channel_similarity_fn(cfg, is_train=None):
    def calc_multi_channel_similarity_fn(h, u=None, mask=None, scope=None):
        if u is not None:
            N, J1, d_h = tf.shape(h)[0], tf.shape(h)[1], tf.shape(h)[2]
            J2 = tf.shape(u)[1]
            aug_h = tf.tile(tf.expand_dims(h, 2), [1, 1, J2, 1]) # [N, J1, J2, d]
            aug_u = tf.tile(tf.expand_dims(u, 1), [1, J1, 1, 1]) # [N, J1, J2, d]
            out = get_multi_channel_logits([aug_h, aug_u], None, True, scope=scope, mask=mask, wd=cfg.wd, 
                                            input_keep_prob=cfg.input_keep_prob, is_train=is_train)
        else:
            out = get_multi_channel_logits([h], None, True, True, scope=scope, mask=mask, wd=cfg.wd, 
                                            input_keep_prob=cfg.input_keep_prob, is_train=is_train)
        return out

    return calc_multi_channel_similarity_fn

def create_calc_multi_perspective_similarity_fn(cfg, is_train=None):
    # Full-Matching
    # Maxpooling-Matching
    # Attentive-Matching
    # Max-Attentive-Matching
    calc_similarity_fn = create_calc_similarity_fn(cfg, is_train=is_train)
    compose_fn = create_compose_fn(cfg)
    def match_fn(h, u, num_perspectives=4, scope=None):
        # h [N, JX, d]
        # u [N, JQ, d]
        with tf.variable_scope(scope or 'match'):
            N, JX, JQ = tf.shape(h)[0], tf.shape(h)[1], tf.shape(u)[1]
            d = h.get_shape().as_list()[-1]
            l = num_perspectives
            aug_h = tf.tile(tf.expand_dims(tf.expand_dims(h, 2), 2), [1, 1, JQ, l, 1]) # [N, JX, JQ, l, d]
            aug_u = tf.tile(tf.expand_dims(tf.expand_dims(u, 2), 1), [1, JX, 1, l, 1]) # [N, JX, JQ, l, d]
            W = tf.get_variable('W', [1, 1, 1, l, d], 'float')
            W = tf.tile(W, [N, JX, JQ, 1, 1])
            aug_h = compose_fn(aug_h, W) # [N, JX, JQ, l, d]
            aug_u = compose_fn(aug_u, W) # [N, JX, JQ, l, d]
            h_u_dot = tf.reduce_sum(aug_h * aug_u, -1)
            h_norm = tf.norm(aug_h, axis=-1)
            u_norm = tf.norm(aug_u, axis=-1)
            out = tf.divide(h_u_dot, h_norm * u_norm) # [N, JX, JQ, l]
            return  out

    def calc_multi_perspective_similarity_fn(h, u, u_f, h_mask, u_mask, num_perspectives=2, keep_rate=1.0, scope=None):
        with tf.variable_scope(scope or 'multi_perspective'):
            N, JX, JQ = tf.shape(h)[0], tf.shape(h)[1], tf.shape(u)[1]
            d = h.get_shape().as_list()[-1]
            l = num_perspectives
            h_u_mask = tf.logical_and(tf.tile(tf.expand_dims(h_mask, -1), [1, 1, JQ]), 
                                        tf.tile(tf.expand_dims(u_mask, 1), [1, JX, 1])) # [N, JX, JQ]
            h1 = match_fn(h, tf.expand_dims(u_f, 1), num_perspectives=num_perspectives, scope='h1')
            h1 = tf.reshape(h1, [N, JX, l])
            h2 = match_fn(h, u, num_perspectives=num_perspectives, scope='h2') # [N, JX, JQ, l]
            h2 = tf.reduce_max(exp_mask(h2, tf.tile(tf.expand_dims(h_u_mask, 3), [1, 1, 1, l])), 2) # [N, JX, l]
            h_u_similarity = calc_similarity_fn(h, u, logit_type='dot', scope='h_u_similarity') # [N, JX, JQ]
            aug_u = tf.tile(tf.expand_dims(u, 1), [1, JX, 1, 1]) # [N, JX, JQ, d]
            u_mean = softsel(aug_u, h_u_similarity, mask=h_u_mask, scope='u_mean') # [N, JX, d]
            h3 = match_fn(tf.reshape(h, [-1, 1, d]), tf.reshape(u_mean, [-1, 1, d]), num_perspectives=num_perspectives, scope='h3')
            h3 = tf.reshape(h3, [N, JX, l])
            max_h_u_similarity = tf.argmax(h_u_similarity, axis=2) # [N, JX, 1]
            max_h_u_similarity = tf.one_hot(max_h_u_similarity, JQ, dtype='float') # [N, JX, JQ]
            u_max_mean = tf.reduce_sum(tf.tile(tf.expand_dims(max_h_u_similarity, 3), [1, 1, 1, d]) * aug_u, 2) # [N, JX, d]
            h4 = match_fn(tf.reshape(h, [-1, 1, d]), tf.reshape(u_mean, [-1, 1, d]), num_perspectives=num_perspectives, scope='h4')
            h4 = tf.reshape(h4, [N, JX, l])
            out = tf.concat([h1, h2, h3, h4], 2) # [N, JX, 4*l]
            return out

    return calc_multi_perspective_similarity_fn

def create_calc_multi_head_similarity_fn(cfg, is_train=None):
    num_heads = 4
    input_keep_prob = cfg.input_keep_prob
    is_train = is_train
    def calc_multi_head_similarity_fn(h, u, h_mask, u_mask, scope=None, num_units=None):
        if num_units == None:
            num_units = h.get_shape().as_list()[-1]
        # u_mask = tf.to_float(u_mask)
        # h_mask = tf.to_float(h_mask)
        with tf.variable_scope(scope or 'multi_head_attention'):
            # Linear projections
            Q = F(u, num_units, activation=tf.identity, input_keep_prob=input_keep_prob, wd=cfg.wd, is_train=is_train, scope='Q') # (N, T_q, C)
            K = F(h, num_units, activation=tf.identity, input_keep_prob=input_keep_prob, wd=cfg.wd, is_train=is_train, scope='K') # (N, T_k, C)
            V = F(h, num_units, activation=tf.identity, input_keep_prob=input_keep_prob, wd=cfg.wd, is_train=is_train, scope='V') # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            aug_V = tf.tile(tf.expand_dims(V_, 1), [1, tf.shape(Q_)[1], 1, 1]) # [N*h, T_q, T_k, C/h]
            mask = tf.tile(tf.expand_dims(h_mask, 1), [num_heads, tf.shape(Q_)[1], 1]) \
                        & tf.tile(tf.expand_dims(u_mask, 2), [num_heads, 1, tf.shape(K_)[1]]) # [N*h, T_q, T_k]
            outputs = softsel(aug_V, outputs, mask=mask)

            # Key Masking
            # key_masks = tf.sign(tf.abs(u_mask, axis=-1)) # (N, T_k)
            
            #key_masks = tf.tile(h_mask, [num_heads, 1]) # (h*N, T_k)
            #key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(h)[1], 1]) # (h*N, T_q, T_k)
            
            #paddings = tf.ones_like(outputs)*(-2**32+1)
            #outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
            # outputs = exp_mask(outputs, key_masks, scope='outputs')

            #outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
             
            # Query Masking
            # query_masks = tf.sign(tf.abs(h_mask, axis=-1)) # (N, T_q)
            #query_masks = tf.tile(u_mask, [num_heads, 1]) # (h*N, T_q)
            #query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(h_mask)[1]]) # (h*N, T_q, T_k)
            #outputs *= query_masks # broadcasting. (N, T_q, C)
              
            # Dropouts
            #outputs = tf.layers.dropout(outputs, rate=1-keep_rate, training=is_train)
                   
            # Weighted sum
            #outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                  
            # Residual connection
            # tf.assert_equal(tf.shape(outputs)[1], tf.shape(h)[1])
            #outputs = outputs + h
            
            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)
            return outputs

    return calc_multi_head_similarity_fn

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

class CacheCell(rnn_cell_impl.RNNCell):
    def __init__(self, 
                num_units,
                num_groups, 
                activation=None,
                reuse=None,
                kernel_initializer=None,
                bias_initializer=None, 
                state_is_tuple=True):
        super(rnn_cell_impl.RNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_groups = num_groups
        if self._num_units % self._num_groups != 0:
            raise ValueError("invalid num_groups, because num_units mod num_groups is not zero,\
                             {} % {} != 0".format(self._num_units, self._num_groups))
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, input, state):
        """Cached LSTM
            input, [N, dw]
            state, [N, k*d]
        """
        sigmoid = tf.nn.sigmoid
        if self._state_is_tuple:
            state, c = state
        N = tf.shape(state)[0]
        k = self._num_groups
        d = self._num_units / k
        state = list(tf.split(state, k, axis=1)) # [N*d, N*d, ,,, N*d] k item in total
        ro = rnn_cell_impl._linear([input] + state, 3*k*d, True, self._bias_initializer, self._kernel_initializer) # [N, 3*k*d]
        r, o, c = tf.split(ro, 3, axis=1) # [N, k*d]
        r = sigmoid(r) # [N, k*d]
        r = tf.add(tf.divide(tf.reshape(r, [N, k, d]), k), \
                    tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, 1, delta=1.0/float(k), dtype=tf.float32), 0), 2), [N, 1, d]))
        # r = tf.add(tf.divide(tf.reshape(r, [N, k, d]), k), tf.range(0, 1, delta=1.0/float(k), dtype=tf.float32))
        r = tf.reshape(r, [N, k*d])
        o = sigmoid(o)
        c_ = self._activation(c)
        new_c = (1 - r) * c + r * c_
        new_state = self._activation(new_c) * o
        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_state, new_c)
        return new_state, new_state


class SRUCell(rnn_cell_impl.RNNCell):
    """docstring for SRUCell"""
    def __init__(self, 
                num_units, 
                activation=None, 
                reuse=None, 
                kernel_initializer=None, 
                bias_initializer=None, 
                state_is_tuple=True):
        # super(rnn_cell_impl.RNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._state_is_tuple = state_is_tuple
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, input, state):
        d = self._num_units
        sigmoid = tf.nn.sigmoid
        if self._state_is_tuple:
            c_tm1, h_tm1 = state
        with tf.variable_scope('input'):
            input_ = rnn_cell_impl._linear(input, d, False, self._bias_initializer, self._kernel_initializer)
        with tf.variable_scope('fr'):
            fr = rnn_cell_impl._linear(input_, 2*d, True, self._bias_initializer, self._kernel_initializer)
        fr = sigmoid(fr)
        f, r = tf.split(fr, 2, axis=1) # [N, d]
        c_t = f * c_tm1 + (1 - f) * input
        h_t = r * self._activation(c_t) + (1 - r_t) * input_
        if self._state_is_tuple:
            new_state = LSTMStateTuple(c_t, h_t)
        return h_t, new_state

class MultiWeightCell(rnn_cell_impl.RNNCell):
    """docstring for MultiWeightCell"""
    def __init__(self, 
                num_units,
                num_weights=3,  
                activation=None, 
                reuse=None, 
                kernel_initializer=None, 
                bias_initializer=None, 
                state_is_tuple=True):
        # super(rnn_cell_impl.RNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_weights = num_weights
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units) 
            if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, input, state):
        sigmoid = tf.nn.sigmoid
        k = self._num_weights
        if self._state_is_tuple:
            c_tm1, h_tm1 = state
        with tf.variable_scope("'ifo'"):
            ifo = rnn_cell_impl._linear([input, h_tm1], 3*d, True, self._bias_initializer, self._kernel_initializer)
        ifo = sigmoid(ifo)
        i, f, o = tf.split(ifo, num_or_size_splits=3, axis=1) # [N, d] x 3
        with tf.variable_scope('c_'):
            c_ = self._activation(rnn_cell_impl._linear([input, h_tm1], k * d, True, self._bias_initializer, self._kernel_initializer)) # [N, kd]
        c_ = tf.reshape(c_, [-1, k, d])
        with tf.variable_scope('p_t'):
            p_t = rnn_cell_impl._linear([input, state], k * d, True, self._bias_initializer, self._kernel_initializer)
        p_t = tf.reshape(p_t, [-1, k, d])
        p_t = tf.nn.softmax(p_t, 1) # [N, k, d]
        c_t = f * c_tm1 + tf.reduce_sum(p_t * c_, 1) # [N, d]
        h_t = o * c_t
        if self._state_is_tuple:
            new_state = LSTMStateTuple(c_t, h_t)
        return h_t, new_state

def cache_lstm(rnn_type, inputs, length, hidden_size, num_groups, layer_num=1, 
                dropout_keep_prob=None, concat=True, initial_state=None, 
                kernel_initializer=tf.random_normal_initializer(stddev=0.1), is_train=False):
    if not rnn_type.endswith('clstm'):
        raise ValueError("")

    if not rnn_type.startswith('bi'):
        cell = CacheCell(hidden_size, num_groups)
        if dropout_keep_prob is not None:
            cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=dropout_keep_prob)
        if layer_num > 1:
            cell = tc.rnn.MultiRNNCell([cells]*layer_num, state_is_tuple=True)

        outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32, initial_state=initial_state)

    else:
        cell_fw = CacheCell(hidden_size, num_groups)
        cell_bw = CacheCell(hidden_size, num_groups)
        if dropout_keep_prob is not None:
            cell_fw = SwitchableDropoutWrapper(cell_fw, is_train, input_keep_prob=dropout_keep_prob)
            cell_bw = SwitchableDropoutWrapper(cell_bw, is_train, input_keep_prob=dropout_keep_prob)
        if layer_num > 1:
            cell_bw_fw = tc.rnn.MultiRNNCell([cell_fw]*layer_num, state_is_tuple=True)
            cell_bw = tc.rnn.MultiRNNCell([cell_bw]*layer_num, state_is_tuple=True)

        if initial_state is not None:
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32, 
                initial_state_fw=initial_state[:, :hidden_size], initial_state_bw=initial_state[:, hidden_size:]
            )
        else:
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
            )

        if concat:
            outputs = tf.concat(outputs, axis=2)
            state = tf.concat(state, axis=1)
        else:
            outputs = outputs[0] + outputs[1]
            state = state[0] + state[1]
    return outputs, state


class AttentionCell(rnn_cell_impl.GRUCell):
    """docstring for AttentionCell"""
    def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        inputs, attention_inputs = tf.split(value=inputs, num_or_size_splits=2, axis=1)
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            bias_ones = tf.constant_initializer(1.0, dtype=inputs.dtype)
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            self._gate_linear = rnn_cell_impl._linear(
                                        [inputs, state, attention_inputs * inputs],
                                        2 * self._num_units,
                                        True,
                                        bias_initializer=bias_ones,
                                        kernel_initializer=self._kernel_initializer)

        value = tf.nn.sigmoid(self._gate_linear([inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = rnn_cell_impl._linear(
                                            [inputs, r_state],
                                            self._num_units,
                                            True,
                                            bias_initializer=self._bias_initializer,
                                            kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def F(inputs, d, activation=tf.nn.relu, kernel_initializer=None, scope=None, use_bias=True, input_keep_prob=1.0, wd=0.0, is_train=None):
    out = dropout(inputs, input_keep_prob, is_train)
    with tf.variable_scope(scope or "projection"):
        out = tf.layers.dense(out, d, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer)
        if wd:
            add_wd(wd)
    return out

class Maxout(object):
    """docstring for Maxout"""
    def __init__(self, dim, num_features=3):
        super(Maxout, self).__init__()
        self.dim = dim
        self.num_features = num_features

    def __call__(self, input, scope=None, input_keep_prob=1.0, wd=0.0, is_train=None):
        N = tf.shape(input)[0]
        with tf.variable_scope(scope or "maxout"):
            out = linear(input, self.num_features * self.dim, False, scope='out')
            out = tf.reshape(out, [N, self.num_features, self.dim])
            out = tf.reduce_max(out, 1) # [N, dim]
            return out

def maxout_layer(input, dim, num_features=3, scope=None, input_keep_prob=1.0, wd=0.0, is_train=None):
    return Maxout(dim, num_features)(input, scope='maxout_layer', input_keep_prob=input_keep_prob, wd=wd, is_train=is_train)

class HyperbolicDistanceScorer(object):
    """docstring for HyperbolicDistanceScorer"""
    def __init__(self):
        super(HyperbolicDistanceScorer, self).__init__()

    def __call__(self, input1, input2, scope=None, input_keep_prob=1.0, wd=0.0, is_train=None, epsilon=1e-8):
        N = tf.shape(input1)[0]
        d1 = input1.get_shape().as_list()[-1]
        d2 = input2.get_shape().as_list()[-1]
        with tf.variable_scope(scope or "hyperbolic_distance"):
            x = 1 + 2 * tf.divide(tf.norm(input1 - input2, axis=-1), \
                        (1 - tf.norm(input1, axis=-1)) * (1 - tf.norm(input2, axis=-1)) + epsilon)
            out = tf.log(x + tf.sqrt(tf.square(x) - 1) + epsilon)
            out = linear(tf.expand_dims(out, 1), 1, True, scope='out')
            return tf.reshape(out, [N])

def hyperbolic_distance_score(input1, input2, scope='hyperbolic_distance_layer', input_keep_prob=1.0, wd=0.0, is_train=None):
    return HyperbolicDistanceScorer()(input1, input2, scope=scope,
                                    input_keep_prob=input_keep_prob, wd=wd, is_train=is_train)

class NeuralTensorLayer(object):
    """docstring for NeuralTensorLayer"""
    def __init__(self, rank=3):
        super(NeuralTensorLayer, self).__init__()
        self.rank = rank
        
    def __call__(self, input1, input2, scope=None, input_keep_prob=1.0, wd=0.0, is_train=None):
        N = tf.shape(input1)[0]
        d1 = input1.get_shape().as_list()[-1]
        d2 = input2.get_shape().as_list()[-1]
        with tf.variable_scope(scope or 'ntn'):
            out1 = tf.reshape(tf.matmul(tf.reshape(linear(input1, self.rank * d2, False, scope='out1', 
                                                            input_keep_prob=input_keep_prob, wd=wd, is_train=is_train), 
                                                    [N, self.rank, d2]), 
                                        tf.expand_dims(input2, -1)), 
                            [N, self.rank]) # [N, rank]
            out2 = linear(tf.concat([input1, input2], -1), self.rank, True, 
                        scope='out2', input_keep_prob=input_keep_prob, wd=wd, is_train=is_train)
            out = tf.nn.tanh(out1 + out2)
            out = linear(out, 1, False, scope='out', input_keep_prob=input_keep_prob, wd=wd, is_train=is_train)
            return out

def neural_tensor_layer(e1, e2, rank=3, scope='ntn_layer', input_keep_prob=1.0, wd=0.0, is_train=None):
    return NeuralTensorLayer(rank)(e1, e2, scope=scope, input_keep_prob=input_keep_prob, wd=wd, is_train=is_train)
