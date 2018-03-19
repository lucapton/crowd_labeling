import numpy as np



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

