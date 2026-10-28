import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):

    d_k = Q.shape[-1]
    K_T = np.swapaxes(K, -1, -2)
    scores = np.matmul(Q, K_T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    context = np.matmul(attention_weights, V)
    return scores, attention_weights, context
