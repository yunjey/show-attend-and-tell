import tensorflow as tf

"""
There are some layer modules for attention based image caption generating.
There are some notations. 
N is batch size.
L is spacial size of feature vector (196)
D is dimension of image feature vector (512)
T is the number of time step which is equal to (length of each caption) - 1.
V is vocabulary size. 
M is dimension of word vector which is embedding size.
H is dimension of hidden state.
"""

def init_lstm(X, W1, b1, W2, b2):
    """
    Inputs:
    - X: mean feature vector of shape (N, D).
    - W1: weights of shape (D, H).
    - b1: biases of shape (H,).
    - W2: weights of shape (D, H).
    - b2: biases of shape (H,).
    Returns:
    - out: output data of shape (N, H).
    """
    h = affine_relu_forward(X, W1, b1)
    # TODO: Drop Out for only h, not out 
    out = affine_tanh_forward(h, W2, b2)
    return out

def word_embedding_forward(X, W_embed):
    """
    Inputs:
    - X: input caption data (contains word index) for entire timeseries of shape (N, T) or single time step of shape (N,).
    - W_embed: embedding matrix of shape (V, M).
    Returns:
    - out: word vector of shape (N, T, M) or (N, M).
    """
    out = tf.nn.embedding_lookup(W_embed, X)
    return out

def encode_feature(X, W):
    """
    Inputs:
    - X: feature vector of shape (N, L, D).
    - W: weights of shape (D, D).
    Returns:
    - out: encoded feature vector of shape (N, L, D).
    """
    L = tf.shape(X)[1]
    D = tf.shape(X)[2]
    
    X = tf.reshape(X, [-1, D])
    out =  tf.matmul(X, W) 
    return tf.reshape(out, [-1, L, D])

def attention_forward(X, X_proj, prev_h, W_proj_h, b_proj, W_att):
    """
    Inputs: 
    - X: feature vector of shape (N, L, D)
    - X_proj: projected feature vector of shape (N, L, D)
    - prev_h: previous hidden state of shape (N, H)
    - W_proj_h: matrix for projecting(or encoding) previous hidden state of shape (H, D)
    - b_proj: biases for projecting of shape (D,)
    - W_att: matrix for hidden-to-out of shape (D, 1)
    Returns:
    - context: output data (context vector) for soft attention of shape (N, D) 
    - alpha: alpha weights for visualization of shape (N, L)
    """
    L = tf.shape(X)[1]
    D = tf.shape(X)[2]

    h_proj = tf.matmul(prev_h, W_proj_h)  # (N, D)
    h_proj = tf.expand_dims(h_proj, 1)   # (N, 1, D)
    hidden = tf.nn.tanh(X_proj + h_proj + b_proj)  # (N, L, D)
    hidden_flat = tf.reshape(hidden, [-1, D])
    out =  tf.matmul(hidden_flat, W_att)  # (N x L, 1)   In this case, we don't need to add bias because of softmax.
    out =  tf.reshape(out, [-1 ,L])
    alpha = tf.nn.softmax(out)   # (N, L)
    alpha_expand = tf.expand_dims(alpha, 2)  # (N, L, 1)
    context = tf.reduce_sum(X * alpha_expand, 1)  # (N, D)
    return context, alpha

def rnn_step_forward(X, prev_h, context, Wx, Wh, Wz, b):
    """
    Inputs:
    - X: word vector for current time step of shape (N, M).
    - context: context vector of shape (N, D)
    - prev_h: previous hidden state of shape (N, H).
    - Wx: matrix for wordvec-to-hidden of shape (M, H).
    - Wh: matrix for  hidden-to-hidden of shape (H, H).
    - Wz: matrix for context-to-hidden of shape(D, H).
    - b: biases of shape (H,).
    Returns:
    - h: hidden states at current time step, of shape (N, H).
    """
    h = tf.nn.tanh(tf.matmul(X, Wx) + tf.matmul(prev_h, Wh) + tf.matmul(context, Wz)) + b
    return h

def gru_step_forward(X, prev_h, context, Wx, Wh, Wz, b, Ux, Uh, Uz, b_u):
    """
    Inputs:
    - X: word vector for current time step of shape (N, M).
    - context: context vector of shape (N, D)
    - prev_h: previous hidden state of shape (N, H).
    - Wx: matrix for wordvec-to-hidden of shape (M, 2H).
    - Wh: matrix for  hidden-to-hidden of shape (H, 2H).
    - Wz: matrix for context-to-hidden of shape(D, 2H).
    - b: biases of shape (2H,).
    - Ux: matrix for wordvec-to-hidden of shape (M, H).
    - Uh: matrix for  hidden-to-hidden of shape (H, H).
    - Uz: matrix for context-to-hidden of shape(D, H).
    - b_u: biases of shape (H,).
    Returns:
    - h: hidden state at current time step, of shape (N, H).
    """
    a = tf.matmul(X, Wx) + tf.matmul(prev_h, Wh) + tf.matmul(context, Wz) + b
    a_z, a_r = tf.split(1, 2, a)
    z = tf.nn.sigmoid(a_z)
    r = tf.nn.sigmoid(a_r)
    h = tf.nn.tanh(tf.matmul(X, Ux) + r * tf.matmul(prev_h, Uh) + tf.matmul(context, Uz) + b_u)
    return (1 - z) * prev_h + z * h

def lstm_step_forward(X, prev_h, prev_c, context, Wx, Wh, Wz, b):
    """
    Inputs:
    - X: word vector for current time step of shape (N, M).
    - context: context vector of shape (N, D)
    - prev_h: previous hidden state of shape (N, H).
    - prev_c: previous cell state of shape (N, H).
    - Wx: matrix for wordvec-to-hidden of shape (M, 4H).
    - Wh: matrix for  hidden-to-hidden of shape (H, 4H).
    - Wz: matrix for context-to-hidden of shape(D, 4H).
    - b: biases of shape (4H,).
    Returns:
    - h: hidden state at current time step, of shape (N, H).
    - c: cell state at current time step, of shape (N, H).
    """

    a = tf.matmul(X, Wx) + tf.matmul(prev_h, Wh) + b #+ tf.matmul(context, Wz) +    
    a_i, a_f, a_o, a_g = tf.split(1, 4, a)
    i = tf.nn.sigmoid(a_i)
    f = tf.nn.sigmoid(a_f)
    o = tf.nn.sigmoid(a_o)
    g = tf.nn.tanh(a_g)

    c = f * prev_c + i * g
    h = o * tf.nn.tanh(c) 
    return h, c
 
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

def affine_sigmoid_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, H).
    - W: weights of shape (H, V).
    - b: biases of shape (V,).
    Returns:
    - out: output data of shape (N, V).
    """
    out = tf.nn.sigmoid(tf.matmul(X, W) + b)
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


def temporal_affine_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, T, H).
    - W: weights of shape (H, V).
    - b: biases of shape (V,).
    Returns:
    - out: Output data of shape (N, T, V).
    """
    T = tf.shape(X)[1]
    H = tf.shape(X)[2]
    V = tf.shape(W)[1]

    X = tf.reshape(X, [-1, H])    
    out = affine_forward(X, W, b)
    return tf.reshape(out, [-1, T, V])

def temporal_affine_relu_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, T, H).
    - W: weights of shape (H, V).
    - b: biases of shape (V,).
    Returns:
    - out: output data of shape (N, T, V).
    """
    out = temporal_affine_forward(X, W, b)
    return tf.nn.relu(out)

def temporal_affine_tanh_forward(X, W, b):
    """
    Inputs:
    - X: input data of shape (N, T, H).
    - W: weights of shape (H, V).
    - b: biases of shape (V,).
    Returns:
    - out: output data of shape (N, T, V).
    """
    out = temporal_affine_forward(X, W, b)
    return tf.nn.tanh(out)

def softmax_loss(X, y, mask):
    """
    Inputs:
    - X: input scores of shape (N, V).
    - y: ground-truth indices of shape (N,) where each element is in the range [0, V).
    - mask: boolean array of shape (N,) where mask[i] tells whether or not the scores at X[i] should contribute to the loss.
    Returns:
    - loss: scalar giving loss
    """
    V =  tf.shape(X)[1]

    y_onehot = tf.cast(tf.one_hot(y, V, on_value=1), tf.float32)
    loss = tf.nn.softmax_cross_entropy_with_logits(X, y_onehot) * tf.cast(mask, tf.float32)
    loss = tf.reduce_sum(loss)
    return loss

def temporal_softmax_loss(X, y, mask):
    """
    Inputs:
    - X: input scores of shape (N, T, V).
    - y: ground-truth indices of shape (N, T) where each element is in the range [0, V).
    - mask: boolean array of shape (N, T) where mask[i, t] tells whether or not the scores at X[i, t] should contribute to the loss.
    Returns:
    - loss: scalar giving loss
    """
    V =  tf.shape(X)[2]

    X = tf.reshape(X, [-1, V])
    y_onehot = tf.cast(tf.one_hot(y, V, on_value=1), tf.float32)
    y_onehot_flat = tf.reshape(y_onehot, [-1, V])
    mask_flat = tf.reshape(mask, [-1])
    loss = tf.nn.softmax_cross_entropy_with_logits(X, y_onehot_flat) * tf.cast(mask_flat, tf.float32)
    loss = tf.reduce_sum(loss)
    return loss