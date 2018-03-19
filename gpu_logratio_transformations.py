from theano import tensor as T
import numpy as np



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
def additive_log_ratio_transform(compositional):
    """Applies the additive log-ratio transform to compositional data."""
    compositional = compositional[:] + eps
    continuous = T.log(compositional[..., :-1] /
                       compositional[..., -1].reshape(compositional.shape[:-1] + (1,)))
    return continuous


# inverse additive log-ratio cl_transform
def inverse_additive_log_ratio_transform(continuous):
    """Inverts the additive log-ratio transform, producing compositional data."""
    compositional = T.stack((T.exp(continuous), T.ones((continuous.shape[0], 1))), axis=continuous.ndim - 1)
    compositional /= compositional.sum(axis=-1, keepdims=1)
    return compositional


# centered log-ratio cl_transform
def centered_log_ratio_transform(compositional):
    """Applies the centered log-ratio transform to compositional data."""
    compositional = compositional[:] + eps
    continuous = T.log(compositional)
    continuous -= continuous.mean(-1, keepdims=True)
    return continuous


# inverse centered log-ratio cl_transform
def inverse_centered_log_ratio_transform(continuous):
    """Inverts the centered log-ratio transform, producing compositional data."""
    return T.nnet.softmax(continuous)


# isometric log-ratio cl_transform
def isometric_log_ratio_transform(compositional, projection_matrix):
    """Applies the isometric log-ratio transform to compositional data."""
    continuous = centered_log_ratio_transform(compositional)
    continuous = T.dot(continuous, projection_matrix)
    return continuous


# inverse isometric log-ratio cl_transform
def inverse_isometric_log_ratio_transform(continuous, projection_matrix):
    """Inverts the isometric log-ratio transform, producing compositional data."""
    continuous = T.dot(continuous, projection_matrix.T)
    compositional = inverse_centered_log_ratio_transform(continuous)
    return compositional

