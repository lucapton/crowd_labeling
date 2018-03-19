import numpy as np


class MV:

    def __init__(self, votes=None, workers=None, instances=None, transform=None, classes=None):

        # data info
        self.V = len(votes)
        self.U = len(np.unique(workers))
        self.I = len(np.unique(instances))
        if classes is not None:
            self.C = len(classes)
        else:
            self.C = len(np.unique(votes))
        self.transform = transform
        self.eps = np.finfo(np.float64).eps

        # info to save
        self.labels = np.zeros((self.I, self.C))

        # estimate label means and covariances using ds
        self.mv(votes, workers, instances)

        # apply transform
        if transform == 'clr':

            def clr(self):
                continuous = np.log(self.labels + self.eps)
                continuous -= continuous.mean(1, keepdims=True)
                return continuous

            self.labels = clr(self)

        elif transform == 'alr':

            def alr(self):
                continuous = np.log(self.labels[:, :-1] / (self.labels[:, -1] + self.eps))
                return continuous

            self.labels = alr(self)

        elif transform == 'ilr':

            # make projection matrix
            self.projectionMatrix = np.zeros((self.C, self.C - 1), dtype=np.float32)
            for it in range(self.C - 1):
                i = it + 1
                self.projectionMatrix[:i, it] = 1. / i
                self.projectionMatrix[i, it] = -1
                self.projectionMatrix[i + 1:, it] = 0
                self.projectionMatrix[:, it] *= np.sqrt(i / (i + 1.))

            def ilr(self):
                continuous = np.log(self.labels + self.eps)
                continuous -= continuous.mean(1, keepdims=True)
                continuous = np.dot(continuous, self.projectionMatrix)
                return continuous

            self.labels = ilr(self)

    # DS optimization using EM
    def mv(self, votes, workers, instances):

        # vote weights
        temp = np.vstack((workers, instances)).T
        temp = np.ascontiguousarray(temp).view(np.dtype((np.void, temp.dtype.itemsize * temp.shape[1])))
        _, unique_counts = np.unique(temp, return_counts=True)
        weights = 1. / unique_counts[instances]

        # initial estimates
        for i in range(self.I):
            ind = instances == i
            for c in range(self.C):
                self.labels[i, c] = ((votes[ind] == c) * weights[ind]).sum()
        self.labels /= self.labels.sum(1, keepdims=True) + self.eps

