# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import warnings
import numpy as np
from GPy.kern import Kern
from GPy.core import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


class Chi2(Kern):
    """
    Chi-square Kernel (Cuturi, Marco. "Positive definite kernels in machine learning." arXiv preprint arXiv:0911.5367 (2009).)
    """
    def __init__(self, input_dim, gamma=1., variance=1., active_dims=None, name='Chi2'):
        """
        Init Chi2.
        :param input_dim: int, number of Input dimensions.
        :param gamma: float, kernel parameter. Must be positive.
        :param active_dims: list of indices on which dimensions this kernel works on, or none if no slicing
        :param name: string, kernel name.
        """
        # init
        super(Chi2, self).__init__(input_dim, active_dims, name)

        # set variance
        if variance <= 0:
            raise ValueError("variance parameter must be positive")
        self.variance = Param('variance', np.asarray([variance]), Logexp())
        self.link_parameter(self.variance)

        # set gamma
        if gamma <= 0:
            raise ValueError("gamma parameter must be positive")
        self.gamma = Param('gamma', np.asarray([gamma]), Logexp())
        self.link_parameter(self.gamma)

    def to_dict(self):
        input_dict = super(Chi2, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.HiK"
        input_dict["gamma"] = self.gamma
        input_dict["variance"] = self.variance
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Chi2(**input_dict)

    def chi2_disatnce(self, X1, X2):
        """
        Compute the Chi-square distance 0.5*\sum( (x_i-x_j)^2/(x_i+x_j) )
        :param X1: array (shape=[n, nb_dims], int or float), input data.
        :param X2: array (shape=[m, nb_dims], int or float), input data.
        :return: array(shape=[n,m], float) Matrix where (i,j) is the chi-squared distance between
        """
        if np.any(X1 < 0) or np.any(X2 < 0):
            raise ValueError('Chi2 kernel requires data to be strictly positive!')
        # compute covariance
        d_mat = np.zeros([X1.shape[0], X2.shape[0]])
        for d in range(self.input_dim):
            column_1 = X1[:, d].reshape(-1, 1)
            column_2 = X2[:, d].reshape(-1, 1)
            num = (column_1 - column_2.T)**2
            den = (column_1 + column_2.T)
            d_mat += np.divide(num, den, out=np.zeros_like(d_mat), where=den != 0)
        return 0.5*d_mat

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        """
        compute covariance matrix
        :param X: array (shape=[n, nb_dims], int or float), input data.
        :param X2: array (shape=[m, nb_dims], int or float), input data.
        :return: array (shape=[n,m], float). Gram matrix, Kij=k(xi,xj).
        """
        # input validation
        if X2 is None:
            X2 = X
        if np.any(X < 0) or np.any(X2 < 0):
            raise ValueError('Chi2 kernel requires data to be strictly positive!')

        # compute covariance
        d_mat = self.chi2_disatnce(X, X2)
        k_mat = self.variance*np.exp(-self.gamma * d_mat)
        return k_mat

    def Kdiag(self, X):
        """
        compute the diagonal of the covariance matrix
        :param X: array (shape=[n,nb_dims], int or float).
        :return: array (shape=[n,_], float). Diagonal values.
        """
        if np.any(X < 0):
            raise ValueError('Chi2 kernel requires data to be strictly positive!')
        return np.ones([np.size(X, axis=0)])*self.variance

    def update_gradients_full(self, dL_dK, X, X2=None):
        """

        :param dL_dK: array (?, float). gradient of objective function w.r.t to K.
        :param X: array (shape=[n,nb_dims], int or float).
        :param X2: array (shape=[m,nb_dims], int or float).
        """
        # input validation
        if X2 is None:
            X2 = X
        if np.any(X < 0) or np.any(X2 < 0):
            raise ValueError('Chi2 kernel requires data to be strictly positive!')

        # compute gradient w.r.t gamma
        d_mat = self.chi2_disatnce(X, X2)
        dgamma = -d_mat**self.variance*np.exp(-self.gamma * d_mat)
        dvar = np.exp(-self.gamma * d_mat)

        # update gradient (Petersen, Kaare Brandt; Pedersen, Michael Syskind. The Matrix Cookbook, Eq. 125)
        # see also: https://gpy.readthedocs.io/en/deploy/tuto_creating_new_kernels.html
        self.gamma.gradient = np.sum(dgamma*dL_dK)
        self.variance.gradient = np.sum(dvar*dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_XX(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_XX_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def input_sensitivity(self, summarize=True):
        return np.ones(self.input_dim)
