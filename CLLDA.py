import numpy as np
try:
    from scipy.stats import mode
except ImportError:
    def mode(a, axis=0):
        scores = np.unique(np.ravel(a))  # get ALL unique values
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)

        for score in scores:
            template = (a == score)
            counts = np.expand_dims(np.sum(template, axis), axis)
            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts
import multiprocessing as mp
from time import time
from copy import deepcopy
from crowd_labeling.logratio_transformations import \
    centered_log_ratio_transform as clr, \
    isometric_log_ratio_transform as ilr, \
    additive_log_ratio_transform as alr, \
    make_projection_matrix as mpm


class CLLDA:
    """
    The :class: CLLDA is a python implementation of Crowd Labeling Latent Dirichlet Allocation.

    This algorithm processes crowd labeling (aka crowd consensus) data where workers label instances
    as pertaining to one or more classes. Allows for calculating resulting label estimates and
    covariances in multiple log-ratio transformed spaces.
    """

    def __init__(self, votes, workers, instances, vote_ids=None, worker_ids=None, instance_ids=None,
                 worker_prior=None, instance_prior=None, transform=None,
                 num_epochs=1000, burn_in=200, updateable=True, save_samples=False, seed=None):

        """
        Initializes settings for the model and automatically calls the inference function.

        :param votes: List of vote values.
        :param workers: List of uses who submitted :param votes.
        :param instances: List of instances to which the :param votes pertain.
        :param vote_ids: (optional) List of vote ids. If provided, :param votes should be a list of integers.
        :param instance_ids: (optional) List of instance ids. If provided, :param instances should be a list of integers.
        :param worker_ids: (optional) List of worker ids. If provided, :param workers should be a list of integers.
        :param worker_prior: (optional) Matrix prior for worker skill (pseudovotes).
        :param instance_prior: (optional) List of class priors (pseudovotes)
        :param transform: log-ratio transform to use.
        :param num_epochs: number of epochs to run for.
        :param burn_in: number of epochs to ignore for convergence of the Gibbs chain.
        :param updateable: If True, will save vote-classes between runs at the expense of memory.
        :param save_samples: option to save vote-classes after each epoch (very memory intensive).
        :param seed: seed for the random number generator if reproducibility is desired.
        """

        # set random seed
        if seed is None:
            seed = np.random.randint(int(1e8))
        self.rng = np.random.RandomState(seed=seed)

        # data info and priors
        self.V = len(votes)
        self.U = len(np.unique(workers))
        self.I = len(np.unique(instances))
        if worker_prior is not None:
            self.worker_prior = np.array(worker_prior)
            if self.worker_prior.ndim == 2:
                self.worker_prior = np.tile(self.worker_prior[np.newaxis, :, :], [self.U, 1, 1])
            self.C = self.worker_prior.shape[1]
            self.R = self.worker_prior.shape[2]
        else:
            if vote_ids is not None:
                self.C = len(vote_ids)
                self.R = self.C
            else:
                self.R = len(np.unique(votes))
                self.C = self.R
            self.worker_prior = (np.eye(self.R) + np.ones((self.R, self.R)) / self.R) * 3
            self.worker_prior = np.tile(self.worker_prior[np.newaxis, :, :], [self.U, 1, 1])
        if instance_prior is None:
            self.instance_prior = np.ones(self.C) / self.C / 4
        else:
            self.instance_prior = instance_prior

        # determine vote IDs
        if vote_ids is None:
            self.vote_ids = np.unique(votes)
            vote_dict = {y: x for x, y in enumerate(self.vote_ids)}
            votes = np.array([vote_dict[x] for x in votes])
        else:
            self.vote_ids = vote_ids
        # determine instance IDs
        if instance_ids is None:
            self.instance_ids = np.unique(instances)
            instance_dict = {y: x for x, y in enumerate(self.instance_ids)}
            instances = np.array([instance_dict[x] for x in instances])
        else:
            self.instance_ids = instance_ids
        # determine worker IDs
        if worker_ids is None:
            self.worker_ids = np.unique(workers)
            worker_dict = {y: x for x, y in enumerate(self.worker_ids)}
            workers = np.array([worker_dict[x] for x in workers])
        else:
            self.worker_ids = worker_ids

        # cl_transform info
        if not isinstance(transform, str) and hasattr(transform, '__iter__'):
            self.transform = tuple(transform)
        else:
            self.transform = (transform,)

        # Gibbs sampling parameters
        self.num_epochs = num_epochs
        self.burn_in = burn_in
        self.num_samples = num_epochs - burn_in

        # info to save
        self.LL = np.nan * np.ones(self.num_epochs)
        self.worker_mats = np.zeros((self.U, self.C, self.R))
        self.labels, self.labels_cov = list(), list()
        for transform in self.transform:
            if transform in (None, 'none', 'clr'):
                self.labels.append(np.zeros((self.I, self.C)))
                self.labels_cov.append(np.zeros((self.I, self.C, self.C)))
            elif transform in ('alr', 'ilr'):
                self.labels.append(np.zeros((self.I, self.C - 1)))
                self.labels_cov.append(np.zeros((self.I, self.C - 1, self.C - 1)))
            else:
                raise Exception('Unknown transform!')
        if save_samples:
            self.samples = np.zeros((self.num_epochs - self.burn_in, self.I, self.C - 1))
        self.updateable = updateable
        self.vote_classes = None

        # estimate label means and covariances using cllda
        self.cllda(votes, workers, instances)

        # clean up
        if not self.updateable:
            self.vote_classes = None
        else:
            self.votes = votes
            self.instances = instances
            self.workers = workers

    # CLLDA optimization using Gibbs sampling
    def cllda(self, votes, workers, instances, starting_epoch=0):
        """
        Performs inference on the :class: CLLDA model.

        :param votes: List of vote values.
        :param workers: List of workers who submitted :param votes.
        :param instances: List of instances to which the :param votes pertain.
        :param starting_epoch: How many epochs have already been incorporated in the averages.
        """

        # precalculate
        worker_prior_sum = self.worker_prior.sum(axis=2)
        instance_prior_sum = self.instance_prior.sum()

        # initial estimates
        if self.vote_classes is None:
            if self.C == self.R:
                self.vote_classes = votes.copy()
            else:
                self.vote_classes = self.rng.randint(0, self.C, self.V)

        # calculate vote weights
        temp = np.vstack((workers, instances)).T
        temp = np.ascontiguousarray(temp).view(np.dtype((np.void, temp.dtype.itemsize * temp.shape[1])))
        _, unique_counts = np.unique(temp, return_counts=True)
        weights = 1. / unique_counts[instances]  # type: np.ndarray

        # initial counts
        counts_across_images = np.zeros(shape=(self.U, self.C, self.R))
        counts_across_workers_and_votes = np.zeros(shape=(self.I, self.C))
        for it_v in range(self.V):
            counts_across_images[workers[it_v], self.vote_classes[it_v], votes[it_v]] += weights[it_v]
            counts_across_workers_and_votes[instances[it_v], self.vote_classes[it_v]] += weights[it_v]
        counts_across_images_and_votes = counts_across_images.sum(axis=2)

        # set cl_transform
        transform = list()
        for tfm in self.transform:
            if tfm in (None, 'none'):
                transform.append(self.identity)
            elif tfm == 'clr':
                transform.append(clr)
            elif tfm == 'alr':
                transform.append(alr)
            elif tfm == 'ilr':
                transform.append(lambda comp: ilr(comp, mpm(self.C)))

        # LDA functions
        def get_data_like():
            like = np.zeros(self.V)
            for it_v in range(self.V):
                i = instances[it_v]
                k = self.vote_classes[it_v]
                u = workers[it_v]
                v = votes[it_v]
                w = weights[it_v]  # type: np.ndarray
                like[it_v] = (counts_across_workers_and_votes[i, k] - w + self.instance_prior[k]) \
                             * (counts_across_images[u, k, v] - w + self.worker_prior[u, k, v]) \
                             / (counts_across_images_and_votes[u, k] - w + worker_prior_sum[u, k])
            return np.log(like).sum()

        def get_label_prob():
            like = (counts_across_workers_and_votes[i, :] + self.instance_prior[:]) \
                   * (counts_across_images[u, :, v] + self.worker_prior[u, :, v]) \
                   / (counts_across_images_and_votes[u, :] + worker_prior_sum[u, :])
            return like / like.sum()

        def update_labels():
            # create update
            numerator = counts_across_workers_and_votes + self.instance_prior
            denominator = counts_across_workers_and_votes.sum(axis=1) + instance_prior_sum
            update = numerator / denominator[:, np.newaxis]
            for it, tfm in enumerate(transform):
                tfmupdate = tfm(update)
                if hasattr(self, 'samples'):
                    self.samples[ep - self.burn_in, :, :] = tfmupdate
                # update labels
                delta = (tfmupdate - self.labels[it]) / (ep - self.burn_in + 1)
                self.labels[it] += delta
                # update labels_M2
                delta_cov = delta[:, :, np.newaxis] * delta[:, :, np.newaxis].transpose(0, 2, 1)
                self.labels_cov[it] += (ep - self.burn_in) * delta_cov - self.labels_cov[it] / (ep - self.burn_in + 1)

        def update_worker_mats():
            # create update
            numerator = counts_across_images + self.worker_prior
            denominator = counts_across_images.sum(axis=2) + worker_prior_sum
            update = numerator / denominator[:, :, np.newaxis]
            # update labels
            delta = (update - self.worker_mats) / (ep - self.burn_in + 1)
            self.worker_mats += delta

        # CLLDA
        start = time()
        for ep in range(starting_epoch, starting_epoch + self.num_epochs):
            # begin epoch
            print('starting epoch ' + str(ep + 1))
            if ep > starting_epoch:
                time_to_go = (time() - start) * (self.num_epochs - ep) / ep
                if time_to_go >= 3600:
                    print('Estimated time to finish: %.2f hours' % (time_to_go / 3600,))
                elif time_to_go >= 60:
                    print('Estimated time to finish: %.1f minutes' % (time_to_go / 60,))
                else:
                    print('Estimated time to finish: %.1f seconds' % (time_to_go,))
            ep_start = time()

            # gibbs sampling
            for it_v in self.rng.permutation(self.V).astype(np.int64):
                # get correct indices
                i = instances[it_v]
                k = self.vote_classes[it_v]
                u = workers[it_v]
                v = votes[it_v]
                w = weights[it_v]
                # decrement counts
                counts_across_images[u, k, v] -= w
                counts_across_workers_and_votes[i, k] -= w
                counts_across_images_and_votes[u, k] -= w
                # calculate probabilities of labels for this vote
                probs = get_label_prob()
                # sample new label
                k = self.rng.multinomial(1, probs).argmax()
                self.vote_classes[it_v] = k
                # increment counts
                counts_across_images[u, k, v] += w
                counts_across_workers_and_votes[i, k] += w
                counts_across_images_and_votes[u, k] += w

            # save information
            self.LL[ep] = get_data_like()
            if ep >= self.burn_in + starting_epoch:
                update_labels()
                update_worker_mats()

            # print epoch LL and duration
            print('Epoch completed in %.1f seconds' % (time() - ep_start,))
            print('LL: %.6f' % (self.LL[ep]))

        # adjust label covariances
        self.labels_cov = [x * self.num_samples / (self.num_samples - 1.) for x in self.labels_cov]

        time_total = time() - start
        if time_total >= 3600:
            print('CLLDA completed in %.2f hours' % (time_total / 3600,))
        elif time_total >= 60:
            print('CLLDA completed in %.1f minutes' % (time_total / 60,))
        else:
            print('CLLDA completed in %.1f seconds' % (time_total,))

    #
    def update(self, votes, workers, instances, vote_ids=None, instance_ids=None, worker_ids=None, worker_prior=None,
               num_epochs=1000, burn_in=200):

        # check that this is updateble
        assert self.updateable, 'This model is not updateable, presumable to conserve memory.'

        # determine IDs
        # for votes
        old_vote_ids = self.vote_ids.copy()  # type: np.ndarray
        if vote_ids is None:
            self.vote_ids = np.unique(votes)
            vote_dict = {y: x for x, y in enumerate(self.vote_ids)}
            votes = np.array([vote_dict[x] for x in votes])
        else:
            self.vote_ids = vote_ids
        # for instances
        old_instance_ids = self.instance_ids.copy()  # type: np.ndarray
        if instance_ids is None:
            self.instance_ids = np.unique(instances)
            instance_dict = {y: x for x, y in enumerate(self.instance_ids)}
            instances = np.array([instance_dict[x] for x in instances])
        else:
            self.instance_ids = instance_ids
        # for workers
        old_worker_ids = self.worker_ids.copy()  # type: np.ndarray
        if worker_ids is None:
            self.worker_ids = np.unique(workers)
            worker_dict = {y: x for x, y in enumerate(self.worker_ids)}
            workers = np.array([worker_dict[x] for x in workers])
        else:
            self.worker_ids = worker_ids

        # update parameters
        self.V = len(votes)
        self.U = len(np.unique(workers))
        self.I = len(np.unique(instances))
        self.num_epochs = num_epochs
        self.burn_in = burn_in

        # add more samples to previous solution
        if np.array_equal(votes, self.votes) and np.array_equal(workers, self.workers) \
                and np.array_equal(instances, self.instances) and np.array_equal(self.vote_ids, old_vote_ids) \
                and np.array_equal(self.instance_ids, old_instance_ids) \
                and np.array_equal(self.worker_ids, old_worker_ids):
            # adjust label covariances
            self.labels_cov = [x * (self.num_samples - 1.) / self.num_samples for x in self.labels_cov]

            # update parameters
            self.LL = np.concatenate((self.LL, np.zeros(num_epochs)))
            old_num_samples = self.num_samples
            self.num_samples += num_epochs - burn_in
            self.votes = votes
            self.workers = workers
            self.instances = instances

            # update cllda
            self.cllda(votes, workers, instances, old_num_samples - 1)

        # keep only vote-classes and build off of them
        else:
            # insert old vote-classes and initialize new vote-classes
            old_vote_classes = self.vote_classes.copy()
            self.vote_classes = np.zeros_like(votes)
            old_dict = {y: x for x, y in enumerate(zip(self.votes, self.workers, self.instances))}
            for it, index in enumerate(zip(votes, workers, instances)):
                try:
                    self.vote_classes[it] = old_vote_classes[old_dict[index]]
                except KeyError:
                    if self.C == self.R:
                        self.vote_classes[it] = votes[it]
                    else:
                        self.vote_classes[it] = self.rng.randint(self.C)

            # adjust worker_prior if necessary
            if not np.array_equal(self.worker_ids, old_worker_ids):
                assert worker_prior is not None, "Worker priors must be provided if worker_ids change."
                self.worker_prior = np.array(worker_prior)
                if self.worker_prior.ndim == 2:
                    self.worker_prior = np.tile(self.worker_prior[np.newaxis, :, :], [self.U, 1, 1])

            # adjust info to save
            self.worker_mats = np.zeros((self.U, self.C, self.R))
            self.labels, self.labels_cov = list(), list()
            for transform in self.transform:
                if transform in (None, 'none', 'clr'):
                    self.labels.append(np.zeros((self.I, self.C)))
                    self.labels_cov.append(np.zeros((self.I, self.C, self.C)))
                elif transform in ('alr', 'ilr'):
                    self.labels.append(np.zeros((self.I, self.C - 1)))
                    self.labels_cov.append(np.zeros((self.I, self.C - 1, self.C - 1)))
                else:
                    raise Exception('Unknown transform!')

            # update parameters
            self.LL = np.zeros(num_epochs)
            self.num_samples = num_epochs

            # update cllda
            self.cllda(votes, workers, instances)
            self.votes = votes
            self.instances = instances
            self.workers = workers

    # no cl_transform
    @staticmethod
    def identity(compositional):
        return compositional


def concurrent_cllda(models, votes, workers, instances, nprocs=4, **kwargs):
    """
    Effortless parallelization of multiple CLLDA models.

    :param models: If creating new models, an integer denoting how many models to create.
        Otherwise, a list of existing models to update.
    :param votes: List of vote values.
    :param workers: List of uses who submitted :param votes.
    :param instances: List of instances to which the :param votes pertain.
    :param nprocs: Number of processors to use in the parallel pool.
    :param kwargs: Other possible inputs to either CLLDA.__init__ or CLLDA.update
    :return: List of new or updated CLLDA models.
    """

    # open parallel pool
    print('Starting multiprocessing pool...')
    pool = mp.Pool(processes=nprocs)

    # run CL-LDA
    if isinstance(models, int):
        print('Starting new CL-LDA models in parallel...')
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        kwargs = [deepcopy(kwargs) for x in range(models)]
        for it in range(models):
            kwargs[it]['seed'] = np.random.randint(int(1e8))
        out = pool.map(_new_cllda, [(votes, workers, instances, kwa) for kwa in kwargs])

    elif hasattr(models, '__iter__'):
        print('Updating CL-LDA models in parallel...')
        out = pool.map(_update_cllda, [(model, votes, workers, instances, kwargs) for model in models])

    else:
        pool.close()
        TypeError('Unknown type for input: models.')

    # close parallel pool
    pool.close()
    print('Multiprocessing pool closed.')

    return out


def combine_cllda(models):
    """
    Combine multiple CLLDA instances.
    :param models: List of CLLDA models trained with the same settings.
    :return: CLLDA model which combines the input models.
    """

    # check models are equivalent
    assert np.equal(models[0].V, [model.V for model in models[1:]]).any(), 'Different number of votes!'
    assert np.equal(models[0].U, [model.U for model in models[1:]]).any(), 'Different number of workers!'
    assert np.equal(models[0].I, [model.I for model in models[1:]]).any(), 'Different number of instances!'
    assert np.equal(models[0].C, [model.C for model in models[1:]]).any(), 'Different number of classes!'
    assert np.equal(models[0].R, [model.R for model in models[1:]]).any(), 'Different number of responses!'
    assert np.equal(models[0].worker_prior,
                    [model.worker_prior for model in models[1:]]).any(), 'Different worker priors!'
    assert np.equal(models[0].instance_prior,
                    [model.instance_prior for model in models[1:]]).any(), 'Different instance priors!'
    assert np.all([models[0].transform == model.transform for model in models[1:]]), 'Different transforms!'

    # data info
    out = deepcopy((models[0]))

    # combine label estimates
    out.num_samples = np.sum([model.num_samples for model in models])

    # combine worker estimates
    out.worker_mats = np.sum([model.worker_mats * model.num_samples for model in models],
                             axis=0) / out.num_samples

    if all([x.updateable for x in models]):
        out.vote_classes = mode(np.stack([x.vote_classes for x in models]))[0].flatten()

    # combine labels and label covariances
    for it in range(len(models[0].transform)):
        out.labels[it] = np.sum([model.labels[it] * model.num_samples for model in models], 0) / out.num_samples
        labels_corrmat = [(model.num_samples - 1.) / model.num_samples * model.labels_cov[it]
                          + model.labels[it][..., np.newaxis] * model.labels[it][..., np.newaxis].transpose(0, 2, 1)
                          for model in models]
        out.labels_cov[it] = np.sum([corrmat * model.num_samples for model, corrmat in zip(models, labels_corrmat)],
                                    0) \
                             / out.num_samples - out.labels[it][..., np.newaxis] * out.labels[it][
            ..., np.newaxis].transpose(0, 2, 1)

        # adjust label covariances
        out.labels_cov[it] *= out.num_samples / (out.num_samples - 1.)

    return out


# map function
def _new_cllda(inputs):
    return CLLDA(*inputs[:3], **inputs[3])


# map function
def _update_cllda(inputs):
    inputs[0].update(*inputs[1:4], **inputs[4])
    return inputs[0]


# if __name__ == '__main__':
#     # test suite
#     from DS import DS
#     test_data = DS.test_data()
#     CLLDA(test_data[0], test_data[1], test_data[2], num_epochs=10, burn_in=2, transform=('none', 'alr', 'ilr', 'clr'))
#     cls = concurrent_cllda(4, test_data[0], test_data[1], test_data[2],
#                            num_epochs=10, burn_in=2, transform=('none', 'alr', 'ilr', 'clr'))
#     cl = combine_cllda(cls)
#     a=1
