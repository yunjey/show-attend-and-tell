import tensorflow as tf

"""
This is a implementation for some layers used in image captioning model.
There are some notations. 
N is batch size.
L is spacial size of feature vector (196)
D is dimension of image feature vector (512)
T is the number of time step which is equal to (length of each caption) - 1.
V is vocabulary size. 
M is dimension of word vector which is embedding size.
H is dimension of hidden state.
"""


def word_embedding_forward(captions_in, W_embed):
    """
    Inputs:
    - captions_in: input caption data (word index) for entire timeseries of shape (N, T) or (N,).
    - W_embed: embedding matrix of shape (V, M).
    Returns:
    - out: word vector of shape (N, T, M) or (N, M).
    """
    out = tf.nn.embedding_lookup(W_embed, captions_in)
    return out

def rnn_step_forward_with_attention(X, features, prev_h, params, hyper_params):
    """
    Inputs:
    - X: input data (word vector) for current time step of shape (N, M).
    - features: input data (feature vectors) of shape (N, L, D)
    - prev_h: previous hidden state of shape (N, H).
    - params: dictionary with the following keys:
        - Wx: matrix for input-to-hidden of shape (M, H).
        - Wh: matrix for  hidden-to-hidden of shape (H, H).
        - Wz: matrix for context-to-hidden of shape(D, 4).
        - b: biases of shape (H,).
    Returns:
    - next_h: hidden states for current time step, of shape (N, H).
    """
    Wx = params['Wx']
    Wh = params['Wh']
    Wz = params['Wz']
    b = params['b']
    context, alpha = attention_forward(features, prev_h, params, hyper_params)
    next_h = tf.nn.tanh(tf.matmul(X, Wx) + tf.matmul(prev_h, Wh) + tf.matmul(context, Wz)) + b
    return next_h, alpha

def rnn_forward(X, features, h0, params, hyper_params):
    """
    Inputs:
    - X: input data for the entire timeseries of shape (N, T, M).
    - h0: initial hidden state of shape (N, H).
    - features: input data (feature vectors) of shape (N, L, D)
    - params: dictionary used in rnn_step_forward_with_attention.
    - hyper_params: dictionary with the following key:
        -n_time_step: time step size
    Returns:
    - h: hidden states for the entire timeseries of shape (N, T, H).
    """
    T = hyper_params['n_time_step']
    prev_h = h0
    h_list = []

    for t in range(T):
        next_h, _ = rnn_step_forward_with_attention(X[:,t,:], features, prev_h, params, hyper_params)
        h_list.append(next_h)  # tensor flow doesn't provide item assignment such as h[:,t,:] = next_h
        prev_h = next_h

    h = tf.transpose(tf.pack(h_list), (1, 0, 2))
    return h


def attention_forward(X, prev_h, params, hyper_params):
    """
    Inputs: 
    - X: input data (feature vectors) of shape (N, L, D)
    - prev_h: previous hidden state of shape (N, H)
    - params: dictionary with the following keys:
        - W_proj_x: matrix for projecting(or encoding) feature vector of shape (D, D)
        - W_proj_h: matrix for projecting(or encoding) previous hidden state of shape (H, D)
        - b_proj: biases of shape (D,)
        - W_att: matrix for hidden-to-out of shape (D, 1)
    - hyper_params: dictionary with the following keys:
        - batch_size: mini batch size
        - spacial_size: spacial size of feature vector
        - dim_feature: dimension of feature vector
    Returns:
    - context: output data (context vector) for soft attention of shape (N, D) 
    - alpha: alpha weights for visualization of shape (N, L)
    """
    W_proj_x = params['W_proj_x']
    W_proj_h = params['W_proj_h']
    b_proj = params['b_proj']
    W_att = params['W_att']

    N = hyper_params['batch_size']
    L = hyper_params['spacial_size']
    D = hyper_params['dim_feature']

    X_flat = tf.reshape(X, [N*L,D]) 
    X_proj = tf.matmul(X_flat, W_proj_x)  # (N x L, D)
    X_proj = tf.reshape(X_proj, [N,L,D])  # (N, L, D)
    h_proj = tf.matmul(prev_h, W_proj_h)  # (N, D)
    h_proj = tf.expand_dims(h_proj, 1)   # (N, 1, D)
    hidden = tf.nn.tanh(X_proj + h_proj + b_proj)  # (N, L, D)
    hidden_flat = tf.reshape(hidden, [N*L,D])
    out =  tf.matmul(hidden_flat, W_att) # (N x L, 1)   In this case, we don't need to add bias because of softmax.
    out =  tf.reshape(out, [N,L])
    alpha = tf.nn.softmax(out)  # (N, L)
    alpha_expand = tf.expand_dims(alpha, 2)  # (N, L, 1)
    context = tf.reduce_sum(X * alpha_expand, 1)  # (N, D)
    return context, alpha

def lstm_step_forward_with_attention(X, features, prev_h, prev_c, params, hyper_params):
    """
    Inputs:
    - X: input data (word vector) for current time step of shape (N, M).
    - features: input data (feature vectors) of shape (N, L, D)
    - prev_h: previous hidden state of shape (N, H).
    - prev_c: previous cell state of shape (N, H).
    - params: dictionary with the following keys:
        - Wx: matrix for input-to-hidden of shape (M, 4H).
        - Wh: matrix for  hidden-to-hidden of shape (H, 4H).
        - Wz: matrix for context-to-hidden of shape(D, 4H).
        - b: biases of shape (4H,).
    - hyper_params: dictionary with the following keys:
        - batch_size: mini batch size
        - spacial_size: spacial size of feature vector
        - dim_feature: dimension of feature vector
    Returns:
    - next_h: next hidden state of shape (N, H).
    - next_c: next cell state of shape (N, H).
    - alpha: alpha weights generated by attention layer of shape (N, L)
    """
    Wx = params['Wx']
    Wh = params['Wh']
    Wz = params['Wz']
    b = params['b']

    context, alpha = attention_forward(features, prev_h, params, hyper_params)

    a = tf.matmul(X, Wx) + tf.matmul(prev_h, Wh) + tf.matmul(context, Wz) + b    
    a_i, a_f, a_o, a_g = tf.split(1, 4, a)
    i = tf.nn.sigmoid(a_i)
    f = tf.nn.sigmoid(a_f)
    o = tf.nn.sigmoid(a_o)
    g = tf.nn.tanh(a_g)

    next_c = f * prev_c + i * g
    next_h = o * tf.nn.tanh(next_c) 
    return next_h, next_c, alpha

def lstm_forward(X, features, h0, c0, params, hyper_params):
    """
    Inputs:
    - x: input data (word vectors) of shape (N, T, D).
    - features: input data (feature vectors) of shape (N, L, D).
    - h0: initial hidden state of shape (N, H).
    - params: dictionary used in lstm_step_forward_with_attention.
    - hyper_params: dictionary with the following keys:
        - n_time_step: time step size
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H).
    """
    T = hyper_params['n_time_step']
    prev_h = h0
    prev_c = c0
    h_list = []

    for t in range(T):
        next_h, next_c, _ = lstm_step_forward_with_attention(X[:, t, :], features, prev_h, prev_c, params, hyper_params)
        h_list.append(next_h)  # tensor flow doesn't provide item assignment such as h[:,t,:] = next_h
        prev_h = next_h
        prev_c = next_c
    h = tf.transpose(tf.pack(h_list), (1, 0, 2))
    return h

def affine_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, H).
    - W: weights of shape (H, H).
    - b: biases of shape (H,).
    Returns:
    - out: output data of shape (N, H).
    """
    out =  tf.matmul(X, W) + b
    return out

def affine_relu_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, H).
    - W: weights of shape (H, V).
    - b: biases of shape (V,).
    Returns:
    - out: output data of shape (N, V).
    """
    out = tf.nn.relu(tf.matmul(X, W) + b)
    return out

def affine_tanh_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, D).
    - W: weights of shape (D, H).
    - b: biases of shape (H,).
    Returns:
    - out: output data of shape (N, H).
    """
    out = tf.nn.tanh(tf.matmul(X, W) + b)
    return out

def init_lstm(X, W1, b1, W2, b2):
    """
    Inputs:
    - X: input data of shape (N, D).
    - W1: weights of shape (D, H).
    - b1: biases of shape (H,).
    - W2: weights of shape (D, H).
    - b2: biases of shape (H,).
    Returns:
    - out: output data of shape (N, H).
    """
    h = tf.nn.tanh(tf.matmul(X, W1) + b1)
    out = tf.nn.tanh(tf.matmul(h, W2) + b2)
    return out



def temporal_affine_forward(X, params, hyper_params):
    """
    Inputs:
    - X: input data of shape (N, T, H).
    - W: weights of shape (H, V).
    - b: biases of shape (V,).
    - hyper_params: dictionary with the following keys:
        - batch_size: mini batch size 
        - n_time_step: time step size
        - dim_hidden: dimension of hidden state
        - vocab_size: vocabulary size
    Returns:
    - out: Output data of shape (N, T, V).
    """
    N = hyper_params['batch_size']
    T = hyper_params['n_time_step']
    H = hyper_params['dim_hidden']
    V = hyper_params['vocab_size']
    M = hyper_params['dim_embed']

    X = tf.reshape(X, [N*T,H])
    h = tf.nn.relu(tf.matmul(X, params['W_MLP_embed']) + params['b_MLP_embed'])
    out = tf.matmul(h, params['W_MLP_vocab']) + params['b_MLP_vocab']   
    return tf.reshape(out, [N,T,V])

def temporal_softmax_loss(X, y, mask, hyper_params):
    """
    Inputs:
    - X: input scores of shape (N, T, V).
    - y: ground-truth indices of shape (N, T) where each element is in the range [0, V).
    - mask: boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.
    - hyper_params: dictionary with the following keys:
        - batch_size: mini batch size 
        - n_time_step: time step size
        - dim_hidden: dimension of hidden state
        - vocab_size: vocabulary size
    Returns:
    - loss: Scalar giving loss
    """
    N = hyper_params['batch_size']
    T = hyper_params['n_time_step']
    H = hyper_params['dim_hidden']
    V = hyper_params['vocab_size']

    X = tf.reshape(X, [N*T,V])
    y_onehot = tf.cast(tf.one_hot(y, V, on_value=1), tf.float32)
    y_onehot_flat = tf.reshape(y_onehot, [N*T,V])
    mask_flat = tf.reshape(mask, [N*T])
    loss = tf.nn.softmax_cross_entropy_with_logits(X, y_onehot_flat) * tf.cast(mask_flat, tf.float32)
    loss = tf.reduce_sum(loss)
    return loss