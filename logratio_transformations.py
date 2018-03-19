import numpy as np


# numpy functions

# additive log-ratio cl_transform
def additive_log_ratio_transform(compositional):
    """Applies the additive log-ratio transform to compositional data."""
    compositional = compositional[:] + np.finfo(compositional.dtype).eps
    continuous = np.log(compositional[..., :-1] / compositional[..., -1, np.newaxis])
    return continuous


# inverse additive log-ratio cl_transform
def inverse_additive_log_ratio_transform(continuous):
    """Inverts the additive log-ratio transform, producing compositional data."""
    n = continuous.shape[0]
    compositional = np.hstack((np.exp(continuous), np.ones((n, 1))))
    compositional /= compositional.sum(axis=-1, keepdims=1)
    return compositional


# centered log-ratio cl_transform
def centered_log_ratio_transform(compositional):
    """Applies the centered log-ratio transform to compositional data."""
    continuous = np.log(compositional + np.finfo(compositional.dtype).eps)
    continuous -= continuous.mean(-1, keepdims=True)
    return continuous


# inverse centered log-ratio cl_transform
def inverse_centered_log_ratio_transform(continuous):
    """Inverts the centered log-ratio transform, producing compositional data."""
    compositional = np.exp(continuous)
    compositional /= compositional.sum(axis=-1, keepdims=1)
    return compositional


# isometric log-ratio cl_transform
def isometric_log_ratio_transform(compositional, projection_matrix):
    """Applies the isometric log-ratio transform to compositional data."""
    continuous = centered_log_ratio_transform(compositional)
    continuous = np.dot(continuous, projection_matrix)
    return continuous


# inverse isometric log-ratio cl_transform
def inverse_isometric_log_ratio_transform(continuous, projection_matrix):
    """Inverts the isometric log-ratio transform, producing compositional data."""
    continuous = np.dot(continuous, projection_matrix.T)
    compositional = inverse_centered_log_ratio_transform(continuous)
    return compositional


# isometric log-ratio cl_transform
def easy_isometric_log_ratio_transform(compositional):
    """Applies the isometric log-ratio transform to compositional data."""
    continuous = centered_log_ratio_transform(compositional)
    projection_matrix = make_projection_matrix(continuous.shape[1])
    continuous = np.dot(continuous, projection_matrix)
    return continuous


# inverse isometric log-ratio cl_transform
def easy_inverse_isometric_log_ratio_transform(continuous):
    """Inverts the isometric log-ratio transform, producing compositional data."""
    projection_matrix = make_projection_matrix(continuous.shape[1] + 1)
    continuous = np.dot(continuous, projection_matrix.T)
    compositional = inverse_centered_log_ratio_transform(continuous)
    return compositional


# projection matrix for isometric log-ratio cl_transform
def make_projection_matrix(dimension):
    """Creates the projection matrix for the  the isometric log-ratio transform."""
    projection_matrix = np.zeros((dimension, dimension - 1), dtype=np.float32)
    for it in range(dimension - 1):
        i = it + 1
        projection_matrix[:i, it] = 1. / i
        projection_matrix[i, it] = -1
        projection_matrix[i + 1:, it] = 0
        projection_matrix[:, it] *= np.sqrt(i / (i + 1.))
    return projection_matrix


# theano functions
eps = np.finfo(np.float32).eps


# additive log-ratio cl_transform
def theano_additive_log_ratio_transform(compositional):
    """Applies the additive log-ratio transform to compositional data."""
    from theano import tensor as T
    compositional = compositional[:] + eps
    continuous = T.log(compositional[..., :-1] /
                       compositional[..., -1].reshape(compositional.shape[:-1] + (1,)))
    return continuous


# inverse additive log-ratio cl_transform
def theano_inverse_additive_log_ratio_transform(continuous):
    """Inverts the additive log-ratio transform, producing compositional data."""
    from theano import tensor as T
    compositional = T.stack((T.exp(continuous), T.ones((continuous.shape[0], 1))), axis=continuous.ndim - 1)
    compositional /= compositional.sum(axis=-1, keepdims=1)
    return compositional


# centered log-ratio cl_transform
def theano_centered_log_ratio_transform(compositional):
    """Applies the centered log-ratio transform to compositional data."""
    from theano import tensor as T
    compositional = compositional[:] + eps
    continuous = T.log(compositional)
    continuous -= continuous.mean(-1, keepdims=True)
    return continuous


# inverse centered log-ratio cl_transform
def theano_inverse_centered_log_ratio_transform(continuous):
    """Inverts the centered log-ratio transform, producing compositional data."""
    from theano import tensor as T
    compositional = T.exp(continuous)
    compositional /= compositional.sum(axis=-1, keepdims=1)
    return compositional


# isometric log-ratio cl_transform
def theano_isometric_log_ratio_transform(compositional, projection_matrix):
    """Applies the isometric log-ratio transform to compositional data."""
    from theano import tensor as T
    continuous = theano_centered_log_ratio_transform(compositional)
    continuous = T.dot(continuous, projection_matrix)
    return continuous


# inverse isometric log-ratio cl_transform
def theano_inverse_isometric_log_ratio_transform(continuous, projection_matrix):
    """Inverts the isometric log-ratio transform, producing compositional data."""
    from theano import tensor as T
    continuous = T.dot(continuous, projection_matrix.T)
    compositional = theano_inverse_centered_log_ratio_transform(continuous)
    return compositional


# tensorflow functions

# additive log-ratio cl_transform
def tf_additive_log_ratio_transform(compositional, name='alrt'):
    """Applies the additive log-ratio transform to compositional data."""
    import tensorflow as tf
    compositional = compositional + eps
    continuous = tf.log(compositional[..., :-1] /
                       compositional[..., -1].reshape(compositional.shape[:-1] + (1,)), name=name)
    return continuous


# inverse additive log-ratio cl_transform
def tf_inverse_additive_log_ratio_transform(continuous, name='ialrt'):
    """Inverts the additive log-ratio transform, producing compositional data."""
    import tensorflow as tf
    compositional = tf.stack((tf.exp(continuous), tf.ones((continuous.shape[0], 1))), axis=tf.get_shape(continuous).ndim - 1)
    compositional /= tf.reduce_sum(compositional, axis=-1, keep_dims=True, name=name)
    return compositional


# centered log-ratio cl_transform
def tf_centered_log_ratio_transform(compositional, name='clrt'):
    """Applies the centered log-ratio transform to compositional data."""
    import tensorflow as tf
    compositional = compositional[:] + eps
    continuous = tf.log(compositional)
    continuous -= tf.reduce_mean(continuous, axis=-1, keep_dims=True)
    if name:
        continuous = tf.identity(continuous, name=name)
    return continuous


# inverse centered log-ratio cl_transform
def tf_inverse_centered_log_ratio_transform(continuous, name='iclrt'):
    """Inverts the centered log-ratio transform, producing compositional data."""
    import tensorflow as tf
    compositional = tf.exp(continuous)
    compositional /= tf.reduce_sum(compositional, axis=-1, keep_dims=True)
    if name:
        compositional = tf.identity(compositional, name=name)
    return compositional


# isometric log-ratio cl_transform
def tf_isometric_log_ratio_transform(compositional, projection_matrix, name='ilrt'):
    """Applies the isometric log-ratio transform to compositional data."""
    import tensorflow as tf
    continuous = tf_centered_log_ratio_transform(compositional, name=None)
    continuous = tf.matmul(continuous, projection_matrix, name=name)
    return continuous


# inverse isometric log-ratio cl_transform
def tf_inverse_isometric_log_ratio_transform(continuous, projection_matrix, name='iilrt'):
    """Inverts the isometric log-ratio transform, producing compositional data."""
    import tensorflow as tf
    continuous = tf.matmul(continuous, projection_matrix, transpose_b=True)
    compositional = tf_inverse_centered_log_ratio_transform(continuous, name=None)
    return tf.identity(compositional, name=name)
