import numpy as np
"""
Isotropic Gaussian Tiri RBF Features
"""


class GaussianRBF:

    def __init__(self, mean, variance, K=2, dims=2):
        """
        :param mean: (np.ndarray) mean vector Kxdim
        :param variance: (np.ndarray)  variance vector Kx1
        :param K: number of basis functions
        :param dims: dimension of the input
        """
        assert mean.shape == (K, dims)
        #assert variance.shape == (K, )

        self._K = K
        self._mean = mean
        self._dims = dims
        self._var = variance

    def _compute(self, point):

        """
        Computes a feature vector for the point given
        :param point: np.ndarray (dim)
        :return: feature vector: np.ndarray
        """
        val = []

        for k in range(self._K):
            dif = self._mean[k, :] - point
            val.append(np.exp(-1/2 * np.dot(dif / self._var[k], dif)))
        f = np.asarray(val, order='F')
        f = f#/np.sum(f)
        return f

    def _compute2(self, point):

        """
        Computes a feature vector for the point given
        :param point: np.ndarray (dim)
        :return: feature vector: np.ndarray
        """
        val = []

        for k in range(self._K):
            dif = self._mean[k, :] - point
            val.append(np.exp(-1/2 * np.dot(dif / self._var[k], dif)))
        f = np.asarray(val, order='F')
        f = f/np.sum(f)
        return f

    def __call__(self, x):
        if x.ndim == 2:
            # print('yes')
            return self._compute_feature_matrix(x), 'yes'
        elif x.ndim == 1:
            return self._compute(x)

    def comp(self, x):
        return self._compute(x)
    def _compute_feature_matrix(self, data):
        """
        Computes the feature matrix for the dataset passed
        :param data: np.ndarray with a sample per row
        :return: feature matrix (np.ndarray) with feature vector for each row.
        """
        assert data.shape[1] == self._dims
        features = []
        for x in range(data.shape[0]):
            # print(x)
            features.append(self._compute(data[x, :]))

        return np.asarray(features)

    def number_of_features(self):
        return self._K


def build_features_mch_state(mch_size, n_basis, state_dim):
    """Create RBF for mountain car"""
    # Number of features
    K = n_basis[0] * n_basis[1]
    # Build the features
    positions = np.linspace(mch_size['position'][0], mch_size['position'][1], n_basis[0])
    speeds = np.linspace(mch_size['speed'][0], mch_size['speed'][1], n_basis[1])
    mean_positions, mean_speeds = np.meshgrid(positions, speeds)
    mean = np.hstack((mean_positions.reshape(K, 1), mean_speeds.reshape(K, 1)))

    position_size = mch_size['position'][1] - mch_size['position'][0]
    positions_var = (position_size / (n_basis[0] - 1) / 2) ** 2
    speed_size = mch_size['speed'][1] - mch_size['speed'][0]
    speeds_var = (speed_size / (n_basis[1] - 1) / 2) ** 2
    var = np.array([np.tile(positions_var, K), np.tile(speeds_var, K)]).T

    return GaussianRBF(mean, var, K=K, dims=state_dim)

def build_features_mch_state_rew(mch_size, n_basis, state_dim):
    """Create RBF for mountain car"""
    # Number of features
    K = n_basis[0] #* n_basis[1]
    # Build the features
    positions = np.linspace(mch_size['position'][0], mch_size['position'][1], n_basis[0])
    # speeds = np.linspace(mch_size['speed'][0], mch_size['speed'][1], n_basis[1])
    mean = np.hstack(np.meshgrid(positions)).reshape(K,1)#, speeds)
    # mean = np.hstack((mean_positions.reshape(K, 1), mean_speeds.reshape(K, 1)))

    position_size = mch_size['position'][1] - mch_size['position'][0]
    var = np.tile((position_size / (n_basis[0] - 1) / 2) ** 2, K)
    # speed_size = mch_size['speed'][1] - mch_size['speed'][0]
    # speeds_var = (speed_size / (n_basis[1] - 1) / 2) ** 2
    # var = np.array([np.tile(positions_var, K), np.tile(speeds_var, K)]).T
    print(mean.shape)
    return GaussianRBF(mean, var, K=K, dims=1)

def build_features_reacher(min_dist, expon=2, var_initial = 0.001):
    """Create RBF for gridworld as functions of the state"""
    # assert n_basis % 2 == 1
    # Number of features
    # K = n_basis
    # Build the features
    mean = [0]
    var = [var_initial]
    d = min_dist
    next_mean = mean[-1] + d
    while next_mean < 12:
        mean.append(next_mean)
        d = d*expon
        next_mean = mean[-1]+d
        var_initial = var_initial*expon
        var.append(np.min((var_initial, 3)))
    mean = np.array(mean).reshape((-1,1))
    mean = np.concatenate((np.flip(-mean[1:]),mean))
    var = np.array(var).reshape((-1, 1))
    var = np.concatenate((np.flip(var[1:]), var))

    mean = np.linspace(-10,10,61).reshape((-1,1))
    K = len(mean)
    var = np.tile(0.1,61)
    return GaussianRBF(mean, var, K=K, dims=1)


def build_features_reacher2(gw_size, n_basis, state_dim):
    """Create RBF for gridworld as functions of the state"""
    # assert n_basis % 2 == 1
    # Number of features
    # K = n_basis
    # Build the features
    K = n_basis * n_basis
    x = np.linspace(-gw_size, gw_size, n_basis)
    y = np.linspace(-gw_size, gw_size, n_basis)
    mean_x, mean_y = np.meshgrid(x, y)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1)))
    state_var = (gw_size / (n_basis - 1)) ** 2
    var = np.tile(state_var, (K))
    print(var)
    return GaussianRBF(mean, var, K=K, dims=state_dim)


def build_features_gw_state(gw_size, n_basis, state_dim):
    """Create RBF for gridworld as functions of the state"""
    # Number of features
    K = n_basis[0] * n_basis[1]
    # Build the features
    x = np.linspace(0, gw_size[0], n_basis[0])
    y = np.linspace(0, gw_size[1], n_basis[1])
    mean_x, mean_y = np.meshgrid(x, y)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1)))

    state_var = (gw_size[0] / (n_basis[0] - 1) / 2) ** 2
    var = np.tile(state_var, (K))
    print('yes')
    return GaussianRBF(mean, var, K=K, dims=state_dim)


