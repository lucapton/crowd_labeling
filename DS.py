import numpy as np
from time import time


class DS:

    def __init__(self, votes=None, workers=None, instances=None, test=None,
                 transform=None, classes=None, num_epochs=1000):

        if votes is None or workers is None or instances is None:
            votes, workers, instances, test = self.test_data()

        # data info
        self.V = len(votes)
        self.U = len(np.unique(workers))
        self.I = len(np.unique(instances))
        if classes is not None:
            self.C = len(classes)
        else:
            self.C = len(np.unique(votes))
        self.transform = transform
        self.instance_prior = np.zeros(self.C)
        self.eps = np.finfo(np.float64).eps

        # EM parameters
        self.max_epochs = num_epochs

        # info to save
        self.LL = np.nan * np.ones(self.max_epochs)
        if test is not None:
            self.accuracy = np.nan * np.ones(self.max_epochs)
        self.labels = np.zeros((self.I, self.C))
        # self.labels_cov = np.zeros((self.I, self.C, self.C))
        self.worker_skill = np.zeros((self.U, self.C, self.C))

        # estimate label means and covariances using ds
        self.ds(votes, workers, instances, test)

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


        # # adjust label covariances and inverses
        # self.__estimate_label_cov_with_inv_wishart()
        # self.labels_cov *= (self.num_samples + 1) / self.num_samples

    # EM functions
    def calculate_log_like(self, worker_ind, vote_ind):
        # calculate log-likelihood (2.7 is DS)
        LL = 0
        for i in range(self.I):
            LL += np.log((self.worker_skill[worker_ind[i], :, vote_ind[i]].prod(0) * self.instance_prior).sum())
        LL /= self.I
        return LL

    # estimate the instance classes given the current parameters (these labels are treated as missing data here for EM)
    def e_step(self, worker_ind, vote_ind):
        # estimate instance classes (2.5 is DS)
        for i in range(self.I):
            self.labels[i, :] = self.worker_skill[worker_ind[i], :, vote_ind[i]].prod(0)
        self.labels *= self.instance_prior[np.newaxis, :]
        self.labels /= self.labels.sum(1, keepdims=True) + self.eps

    # update parameters to maximize the data likelihood
    def m_step(self, instance_ind):
        # argmax LL over class probabilities (2.4 is DS)
        self.instance_prior = self.labels.mean(0) + self.eps
        # argmax LL over worker skill (2.3 is DS)
        for u in range(self.U):
            for c in range(self.C):
                self.worker_skill[u, :, c] = self.labels[instance_ind[u][c], :].sum(0)
        self.worker_skill /= self.worker_skill.sum(2, keepdims=True) + self.eps

    # DS optimization using EM
    def ds(self, votes, workers, instances, test):
        # precalculate indices
        print('Generating indices...')
        worker_ind, vote_ind, instance_ind = [], [], []
        for i in range(self.I):
            _instance_ind = instances == i
            worker_ind.append(workers[_instance_ind])
            vote_ind.append(votes[_instance_ind])
        for u in range(self.U):
            instance_ind.append([])
            _worker_ind = workers == u
            for c in range(self.C):
                _vote_ind = votes == c
                instance_ind[u].append(instances[np.bitwise_and(_worker_ind, _vote_ind)])

        # DS
        start = time()
        for ep in range(self.max_epochs):
            # begin epoch
            print('starting epoch ' + str(ep + 1))
            if ep:
                time_to_go = (time() - start) * (self.max_epochs - ep) / ep
                if time_to_go >= 3600:
                    print('Estimated time to finish: %.2f hours' % (time_to_go / 3600,))
                elif time_to_go >= 60:
                    print('Estimated time to finish: %.2f minutes' % (time_to_go / 60,))
                else:
                    print('Estimated time to finish: %.1f seconds' % (time_to_go,))
            ep_start = time()

            # EM
            print('E step...')
            if not ep:
                # initial estimates
                for i in range(self.I):
                    ind = instances == i
                    for c in range(self.C):
                        self.labels[i, c] = np.count_nonzero(votes[ind] == c)
                self.labels /= self.labels.sum(1, keepdims=True) + self.eps
                # self.labels[np.random.choice(self.I, self.I/2, replace=False), :] = 1. / self.C
            else:
                self.e_step(worker_ind, vote_ind)
            print('M step...')
            self.m_step(instance_ind)

            # save information
            print('Calculating log-likelihood...')
            self.LL[ep] = self.calculate_log_like(worker_ind, vote_ind)
            print('Log-likelihood = %f' % (self.LL[ep],))

            # evaulation if test available
            if test is not None:
                self.accuracy[ep] = (self.labels.argmax(1) == test).mean()
                print('Accuracy = %f' % (self.accuracy[ep],))
                # ce = -np.log(self.labels[range(self.I), test] + self.eps).sum()
                # print 'Cross Entropy = %f' % (ce,)

            # print epoch duration
            print('Epoch completed in %.1f seconds' % (time() - ep_start,))

        time_total = time() - start
        if time_total >= 3600:
            print('DS completed in %.2f hours' % (time_total / 3600,))
        elif time_total >= 60:
            print('DS completed in %.2f minutes' % (time_total / 60,))
        else:
            print('DS completed in %.1f seconds' % (time_total,))

    # # generate covariance estimates using inverse Wishart prior
    # def __estimate_label_cov_with_inv_wishart(self):
    #     # prepare parameters
    #     self.inv_wishart_prior_scatter = 0.1 * np.eye(self.C - 1) * self.num_samples
    #     self.inv_wishart_degrees_of_freedom = self.C - 1
    #     scatter_matrix = self.labels_cov * self.num_samples
    #
    #     # calculate multivariate student-t covariance based on normal with known mean and inverse Wishart prior
    #     self.labels_cov_iwp = (self.inv_wishart_prior_scatter + scatter_matrix) \
    #                           / (self.inv_wishart_degrees_of_freedom + self.num_samples - (self.C - 1) - 1)
    #
    #     # calculate covariance inverses for later use
    #     self.labels_icov_iwp = np.linalg.inv(self.labels_cov_iwp )

    @staticmethod
    def test_data():
        """
        Sample data from the Dawid & Skene (1979) paper
        :return: (votes, workers, instances, true_class)
        """
        # data from DS section 4
        data = [[[1, 1, 1], 1, 1, 1, 1],
                [[3, 3, 3], 4, 3, 3, 4],
                [[1, 1, 2], 2, 1, 2, 2],
                [[2, 2, 2], 3, 1, 2, 1],
                [[2, 2, 2], 3, 2, 2, 2],
                [[2, 2, 2], 3, 3, 2, 2],
                [[1, 2, 2], 2, 1, 1, 1],
                [[3, 3, 3], 3, 4, 3, 3],
                [[2, 2, 2], 2, 2, 2, 3],
                [[2, 3, 2], 2, 2, 2, 3],
                [[4, 4, 4], 4, 4, 4, 4],
                [[2, 2, 2], 3, 3, 4, 3],
                [[1, 1, 1], 1, 1, 1, 1],
                [[2, 2, 2], 3, 2, 1, 2],
                [[1, 2, 1], 1, 1, 1, 1],
                [[1, 1, 1], 2, 1, 1, 1],
                [[1, 1, 1], 1, 1, 1, 1],
                [[1, 1, 1], 1, 1, 1, 1],
                [[2, 2, 2], 2, 2, 2, 1],
                [[2, 2, 2], 1, 3, 2, 2],
                [[2, 2, 2], 2, 2, 2, 2],
                [[2, 2, 2], 2, 2, 2, 1],
                [[2, 2, 2], 3, 2, 2, 2],
                [[2, 2, 1], 2, 2, 2, 2],
                [[1, 1, 1], 1, 1, 1, 1],
                [[1, 1, 1], 1, 1, 1, 1],
                [[2, 3, 2], 2, 2, 2, 2],
                [[1, 1, 1], 1, 1, 1, 1],
                [[1, 1, 1], 1, 1, 1, 1],
                [[1, 1, 2], 1, 1, 2, 1],
                [[1, 1, 1], 1, 1, 1, 1],
                [[3, 3, 3], 3, 2, 3, 3],
                [[1, 1, 1], 1, 1, 1, 1],
                [[2, 2, 2], 2, 2, 2, 2],
                [[2, 2, 2], 3, 2, 3, 2],
                [[4, 3, 3], 4, 3, 4, 3],
                [[2, 2, 1], 2, 2, 3, 2],
                [[2, 3, 2], 3, 2, 3, 3],
                [[3, 3, 3], 3, 4, 3, 2],
                [[1, 1, 1], 1, 1, 1, 1],
                [[1, 1, 1], 1, 1, 1, 1],
                [[1, 2, 1], 2, 1, 1, 1],
                [[2, 3, 2], 2, 2, 2, 2],
                [[1, 2, 1], 1, 1, 1, 1],
                [[2, 2, 2], 2, 2, 2, 2]]

        # solutions from DS section 4
        test = [1,
                4,
                2,
                2,
                2,
                2,
                1,
                3,
                2,
                2,
                4,
                3,
                1,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                3,
                1,
                2,
                2,
                4,
                2,
                3,
                3,
                1,
                1,
                1,
                2,
                1,
                2]

        # cl_transform to list format
        votes, workers, instances = [], [], []
        for it_patient, patient in enumerate(data):
            for it_doctor, doctor in enumerate(patient):
                if isinstance(doctor, list):
                    for diagnosis in doctor:
                        votes.append(diagnosis-1)
                        workers.append(it_doctor)
                        instances.append(it_patient)
                else:
                    votes.append(doctor-1)
                    workers.append(it_doctor)
                    instances.append(it_patient)

        return np.array(votes), np.array(workers), np.array(instances), np.array(test) - 1


if __name__ == '__main__':
    ds = DS()
