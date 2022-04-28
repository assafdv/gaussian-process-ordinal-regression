import os
import unittest
from ddt import ddt, data, unpack
import warnings
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
from gp_trainer import GpOrdinalModelTrainer
from tempfile import TemporaryDirectory
logging.getLogger().setLevel('INFO')

TEMP_PATH = '/tmp'


@ddt
class GPOrdinalTests(unittest.TestCase):
    """
    Unit Test for GP Model framework
    """
    def setUp(self):
        # init
        warnings.simplefilter('ignore', category=DeprecationWarning)
        np.random.seed(0)
        self.mae_max = 0.2
        self.f1_min = 0.7

    @staticmethod
    def draw_input_points(nb_samples, nb_dims):
        """
        draw random input points
        :param nb_samples: int, number of samples to generate
        :param nb_dims: int, number of input dimensions.
        :return: array (shape=[n_samples, n_dims], float). input points
        """
        X = np.random.rand(nb_samples, nb_dims)
        return X

    @staticmethod
    def draw_latent_gp(X, kern):
        """
        draw samples from the latent GP
        :param X: array (shape=[n_samples, n_dims], float). input points
        :param kern: GPy kernel object.
        :return: array (shape=[n_samples,_], float). input points.
        """
        nb_samples = np.size(X, axis=0)
        K = kern.K(X)
        f = np.random.multivariate_normal(mean=np.zeros(nb_samples), cov=K).reshape(-1, 1)
        return f

    @staticmethod
    def draw_ordinal_variables(f, lik):
        """
        draw ordinal variables
        :param f: array (shape=[n_sampels,_], float). latent gp values.
        :param lik: GPy Ordinal likelihood
        :return: array (shape=[n_samples,_], int). ordinal variables.
        """
        y = lik.samples(f)
        return y

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        evaluate performance
        :param y_true: array (shape=[n_samples,1], int), observed (ordinal) variables .
        :param y_pred: array (shape=[n_samples, 1], int). predicted (ordinal) variables.
        :return: tuple (mae, f1) where mae is a float indicate the mean-absolute-error and f1 is a float indicate the
        f1 score (macro averaged).
        """
        mae = mean_absolute_error(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return mae, f1

    @data((5, True), (5, False), (4, True), (4, False), (3, True), (3, False), (2, True), (2, False))
    @unpack
    def test_model_rbf_kernel(self, nb_ordinal, fix_thresholds):
        """
        test ordinal likelihood model with RBF kernel
        """
        # init
        nb_dims = 1
        n_train = 50
        n_test = 50
        n_samples = n_train + n_test
        logging.info('Ordinal Regression Model Test with RBF kernel and {:d} ordinal variables'.format(nb_ordinal))
        self.model = GpOrdinalModelTrainer(target_labels=np.arange(nb_ordinal),
                                           fix_thresholds=fix_thresholds)

        # set kernel
        k1 = dict()
        k1.setdefault('active_dims', np.arange(nb_dims))
        k1.setdefault('kernel_func', 'rbf')
        k1.setdefault('params', {})
        k1['params'].setdefault('ard', False)
        k1['params'].setdefault('variance', 1)
        k1['params'].setdefault('lengthscale', 0.1)
        kern_params = [k1]

        kern = self.model.create_combined_kernel_obj(kern_params)

        # set likelihood
        lik = self.model.create_ordinal_likelihood(nb_ordinal=nb_ordinal,
                                                   fix_thresholds=fix_thresholds,
                                                   noise_std=1e-6)

        # draw input point
        logging.info('draw input points. n_samples = {:d}, n_dims={:d}'.format(n_samples, nb_dims))
        X = self.draw_input_points(n_samples, nb_dims)

        # draw samples from the latent GP
        logging.info('draw samples from the latent GP (RBF kernel)')
        f = self.draw_latent_gp(X, kern)

        # draw samples from the ordinal likelihood
        logging.info('draw samples from the ordinal likelihood')
        y = self.draw_ordinal_variables(f, lik)

        # split train test
        X_train = X[:n_train, :]
        y_train = y[:n_train, :]

        X_new = X[n_train:, :]
        y_new = y[n_train:, :]

        # train model
        logging.info('train model. n_train = {:d}'.format(n_train))
        self.model.train(X_train, y_train, kern_params=kern_params)

        # predict
        logging.info('predict. n_test = {:d}'.format(n_test))
        y_new_pred,_,_ = self.model.predict(X_new)

        # evaluate
        logging.info('evaluate')
        mae, f1 = self.evaluate(y_new, y_new_pred)
        logging.info('MAE: {:.3}. f1: {:.3}'.format(mae, f1))

        self.assertGreater(self.mae_max, mae)
        self.assertGreater(f1, self.f1_min)

    @data((5, True), (5, False))
    @unpack
    def test_model_chi2_kernel(self, nb_ordinal, fix_thresholds):
        """
        test ordinal likelihood model with RBF kernel
        """
        # init
        nb_dims = 4
        n_train = 350
        n_test = 100
        n_samples = n_train + n_test
        logging.info('Ordinal Regression model test with Chi2 kernel and {:d} ordinal variables'.format(nb_ordinal))
        self.model = GpOrdinalModelTrainer(target_labels=np.arange(nb_ordinal),
                                           fix_thresholds=fix_thresholds)

        # set kernel
        k1 = dict()
        k1.setdefault('active_dims', np.arange(nb_dims))
        k1.setdefault('kernel_func', 'chi2')
        k1.setdefault('params', {})
        k1['params'].setdefault('gamma', 6)
        k1['params'].setdefault('fix_variance', False)
        kern_params = [k1]
        kern = self.model.create_combined_kernel_obj(kern_params)

        # set likelihood
        lik = self.model.create_ordinal_likelihood(nb_ordinal=nb_ordinal,
                                                   fix_thresholds=fix_thresholds,
                                                   noise_std=1e-6)

        # draw input point
        logging.info('draw input points. n_samples = {:d}, n_dims={:d}'.format(n_samples, nb_dims))
        X = self.draw_input_points(n_samples, nb_dims)

        # draw samples from the latent GP
        logging.info('draw samples from the latent GP')
        f = self.draw_latent_gp(X, kern)

        # draw samples from the ordinal likelihood
        logging.info('draw samples from the ordinal likelihood')
        y = self.draw_ordinal_variables(f, lik)
        logging.info('ordinal variables: {}'.format(np.unique(y, return_counts=True)))

        # split train test
        X_train = X[:n_train, :]
        y_train = y[:n_train, :]

        X_new = X[n_train:, :]
        y_new = y[n_train:, :]

        # train model
        logging.info('train model. n_train = {:d}'.format(n_train))
        self.model.train(X_train, y_train, kern_params=kern_params)

        # predict
        logging.info('predict. n_test = {:d}'.format(n_test))
        y_new_pred,_,_ = self.model.predict(X_new)

        # evaluate
        logging.info('evaluate')
        mae, f1 = self.evaluate(y_new, y_new_pred)
        logging.info('MAE: {:.3}. f1: {:.3}'.format(mae, f1))

        self.assertGreater(self.mae_max, mae)
        self.assertGreater(f1, self.f1_min)

    @data((5, True), (5, False))
    @unpack
    def test_model_rbf_chi2_kernel(self, nb_ordinal, fix_thresholds):
        """
        test ordinal likelihood model with RBF kernel
        """
        # init
        nb_dims = 5
        n_train = 450
        n_test = 100
        n_samples = n_train + n_test
        logging.info('Ordinal Regression model test with RBF and Chi2 kernels and {:d} ordinal variables'.format(nb_ordinal))
        self.model = GpOrdinalModelTrainer(target_labels=np.arange(nb_ordinal),
                                           fix_thresholds=fix_thresholds)

        # set kernel
        k1 = dict()
        k1.setdefault('active_dims', np.array([0]))
        k1.setdefault('kernel_func', 'rbf')
        k1.setdefault('params', {})
        k1['params'].setdefault('ard', False)
        k1['params'].setdefault('variance', 1)
        k1['params'].setdefault('lengthscale', 1)
        k2 = dict()
        k2.setdefault('active_dims', np.array([1, 2, 3, 4]))
        k2.setdefault('kernel_func', 'chi2')
        k2.setdefault('params', {})
        k2['params'].setdefault('gamma', 5)
        kern_params = [k1, k2]
        kern = self.model.create_combined_kernel_obj(kern_params)

        # set likelihood
        lik = self.model.create_ordinal_likelihood(nb_ordinal=nb_ordinal,
                                                   fix_thresholds=fix_thresholds,
                                                   noise_std=1e-6)

        # draw input point
        logging.info('draw input points. n_samples = {:d}, n_dims={:d}'.format(n_samples, nb_dims))
        X = self.draw_input_points(n_samples, nb_dims)

        # draw samples from the latent GP
        logging.info('draw samples from the latent GP')
        f = self.draw_latent_gp(X, kern)

        # draw samples from the ordinal likelihood
        logging.info('draw samples from the ordinal likelihood')
        y = self.draw_ordinal_variables(f, lik)
        logging.info('ordinal variables: {}'.format(np.unique(y, return_counts=True)))

        # split train test
        X_train = X[:n_train, :]
        y_train = y[:n_train, :]

        X_new = X[n_train:, :]
        y_new = y[n_train:, :]

        # train model
        logging.info('train model. n_train = {:d}'.format(n_train))
        self.model.train(X_train, y_train, kern_params=kern_params)

        # predict
        logging.info('predict. n_test = {:d}'.format(n_test))
        y_new_pred, _, _ = self.model.predict(X_new)

        # evaluate
        logging.info('evaluate')
        mae, f1 = self.evaluate(y_new, y_new_pred)
        logging.info('MAE: {:.3}. f1: {:.3}'.format(mae, f1))

        self.assertGreater(self.mae_max, mae)
        self.assertGreater(f1, self.f1_min)


if __name__ == '__main__':
    unittest.main(verbosity=2)
