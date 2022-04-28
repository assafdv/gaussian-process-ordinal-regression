import os
import logging
import pickle
import numpy as np
import GPy
from sklearn.preprocessing import LabelEncoder
from gpy_models.ordinal_reg.ordinal_likelihood import Ordinal
from gpy_models.ordinal_reg.gp_ordinal_regression import GPOrdinalRegression
from gpy_models.kernels.histogram_intersection_kernel import HiK
from gpy_models.kernels.chi_square_kernel import Chi2
logging.getLogger().setLevel('INFO')

ORDINAL_NOISE_STD = 1
ORDINAL_DELTA_PRIOR_STD = 0.25

class GpOrdinalModelTrainer:
    """ Gaussian Process Ordinal Regression Model """
    def __init__(self,
                 target_labels,
                 fix_thresholds=False,
                 random_state=0):
        """
        init GPy model.
        :param name: string, model name.
        :param target_labels: array (shape=[n_labels,_]) with class labels for encoding the target variables.
        :param model_file_path: path to existing model file.
        :param random_state: float, random state.
        """
        if not issubclass(target_labels.dtype.type, np.integer):
            raise ValueError("target labels must contains integers")
        self.target_labels = target_labels
        if target_labels.size >2:
            self.fix_thresholds = fix_thresholds
        else:
            self.fix_thresholds = True
        self.label_encoder = LabelEncoder().fit(self.target_labels)
        self.random_state = random_state

    def create_model(self, X, y, kern_params):
        """
        create GPy model
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples, 1], float). Targets.
        :param kern_params: list (dict), each element is a dictionary with kernel parameters per subset of input
        dimensions.
        :return: GPy model object.
        """
        # create kernel
        kern = self.create_combined_kernel_obj(kern_params)

        # create likelihood
        nb_ordinal = len(self.target_labels)
        lik = self.create_ordinal_likelihood(nb_ordinal=nb_ordinal, fix_thresholds=self.fix_thresholds)

        # create Ordinal Regression model
        model = GPOrdinalRegression(X, y, kernel=kern, likelihood=lik)

        # check gradients
        if "DEBUG_MODE" in os.environ:
            model.checkgrad(verbose=True)
        return model

    @staticmethod
    def create_ordinal_likelihood(nb_ordinal,
                                  noise_std=ORDINAL_NOISE_STD,
                                  fix_thresholds=False):
        """
        create GPy likelihood object.
        :param nb_ordinal: int, number of ordinal variables.
        :param sigma: float, noise variance.
        :param fix_thresholds: bool, if True, fix the thresholds parameters in the likelihood.
        :return: GPy likelihood object.
        """
        # compute the bin_edges (the values at which the labels switch)
        bin_edgs = np.arange(nb_ordinal)
        bin_edgs = bin_edgs - np.mean(bin_edgs)
        if nb_ordinal < 1:
            raise ValueError('number of ordinal variable must be at least 1')
        if nb_ordinal <= 2:
            eta1 = bin_edgs[1]
            delta_vals = np.array([])
            delta_vals_prior_mean = None
        else:
            eta1 = bin_edgs[1]
            delta = bin_edgs[2] - bin_edgs[1]
            delta_vals = np.ones(nb_ordinal - 2) * delta
            delta_vals_prior_mean = np.log(delta) - ORDINAL_DELTA_PRIOR_STD**2/2
        lik = Ordinal(eta1=eta1,
                      delta_vals=delta_vals,
                      delta_vals_prior_mean=delta_vals_prior_mean,
                      delta_vals_prior_sigma=ORDINAL_DELTA_PRIOR_STD,
                      sigma=noise_std,
                      fix_thresholds=fix_thresholds)
        return lik

    @staticmethod
    def create_single_kernel_obj(kern_func, kern_name, nb_dims, active_dims, params):
        """
        create GPy kernel object.
        :param kern_func: string, kernel function (e.g. rbf)
        :param kern_name: string, kernel name.
        :param nb_dims: int, number of dimensions.
        :param active_dims: array (shape=[nb_dims,_], int). active dimensions for slicing.
        :param params: dict, kernel params.
        :return: GPy kernel object
        """
        if kern_func == 'linear':
            var = params.get('variance', 1)
            ard = params.get('ard', False)
            if ard:
                variances = np.ones(nb_dims)*var
            else:
                variances = var
            logging.info("Create Linear kernel. dims = {}, variance = {}, ARD = {}.".format(nb_dims, var, ard))
            kern = GPy.kern.Linear(nb_dims, active_dims=active_dims, variances=variances, ARD=ard, name=kern_name)
        elif kern_func == 'rbf':
            var = params.get('variance', 1)
            ls = params.get('lengthscale', 1)
            ard = params.get('ard', False)
            fix_var = params.get('fix_variance', True)
            if ard:
                lengthscale = np.ones(nb_dims)*ls
            else:
                lengthscale = ls
            logging.info("Create RBF kernel. dims = {}, variance = {}, lengthscale = {}, ARD = {}."
                         .format(nb_dims, var, ls, ard))
            kern = GPy.kern.RBF(nb_dims,
                                active_dims=active_dims,
                                variance=var,
                                lengthscale=lengthscale,
                                ARD=ard,
                                name=kern_name)
            if fix_var:
                kern.variance.fix()
        elif kern_func == 'hik':
            logging.info("Create HiK kernel. dims = {}.".format(nb_dims))
            kern = HiK(nb_dims, active_dims=active_dims, name=kern_name)
        elif kern_func == 'chi2':
            gamma = params.get('gamma', 1)
            variance = params.get('variance', 1)
            fix_var = params.get('fix_variance', True)
            logging.info("Create Chi-squared kernel. dims = {}.".format(nb_dims))
            kern = Chi2(nb_dims, active_dims=active_dims, gamma=gamma, variance=variance, name=kern_name)
            if fix_var:
                kern.variance.fix()
        else:
            raise ValueError("unsupported kernel function")
        return kern

    def create_combined_kernel_obj(self, kern_params):
        """
        Create combined kernel object.
        :param kern_params: list (dict), each element is a dictionary with kernel parameters per subset of input
        dimensions.
        :return: GPy kernel object.
        """
        # loop over kernels
        k = None
        for ks in kern_params:
            # get params
            kern_func = ks['kernel_func']
            active_dims = ks['active_dims']
            kern_name = ks.get('kernel_name', None)
            nb_dims = active_dims.size

            # create kernel
            k_i = self.create_single_kernel_obj(kern_func, kern_name, nb_dims, active_dims, ks['params'])

            # aggregate
            if k is None:
                k = k_i
            else:
                k *= k_i
        return k

    def train(self, X, y, kern_params):
        """
        train model
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples,_], float). Targets.
        :param model_selection_method: str, model selection method.
        :param kern_params: list (dict), each element is a dictionary with kernel parameters per subset of input
        dimensions.
        """
        # encode targets
        ye = self.encode_targets(y)
        ye = ye.reshape(-1, 1)

        # create model
        logging.info('Create new GPy model')
        self.model = self.create_model(X, ye, kern_params)

        # optimize model
        self.optimize_model(num_restarts=5)

        if "DEBUG_MODE" in os.environ:
            logging.info(self.model)
        else:
            logging.info('(negative) Marginal log-likelihood: {:.3} '.format(self.model.objective_function()))

    def encode_targets(self, y):
        """
        encode target variables
        :param y: array (shape=[n_samples,_], float). Targets.
        :return: array (shape=[n_samples,_], float). Encoded targets.
        """
        if not np.isin(y, self.label_encoder.classes_).all():
            raise ValueError("targets include unsupported labels")
        ye = self.label_encoder.transform(y)
        return ye

    def decode_targets(self, ye):
        """
        decode target variables
        :param ye: array (shape=[n_samples,_], float). Encoded targets.
        :return: array (shape=[n_samples,_], float). Decoded targets.
        """
        # decode
        try:
            y = self.label_encoder.inverse_transform(ye)
        except Exception as e:
            raise("internal error in decoding targets: " + str(e))
        return y

    def optimize_model(self, n_iter=1, num_restarts=10, max_iter=200):
        """
        model adaptation in Bayesian fashion. Find optimal hyper-parameters that maximize the marginal likelihood.
        :param n_iter: int, number of parameters optimization + posterior approximation runs.
        :param num_restarts: int, number of restarts to use.
        :param max_iter: int, number of iteration for parameters optimization.
        """
        # we interleave runs of posterior approximation with optimization of the parameters using gradient descent
        # methods. Whilst the parameters are being optimized, the posterior approximation is fixed.
        # for i in range(n_iter):
        #     self.model.optimize(max_iters=max_iter, messages=False)
        if "DEBUG_MODE" in os.environ:
            messages = True
        else:
            messages = False

        for i in range(n_iter):
            # optimize kernel params
            lik_params = self.get_obj_params(self.model.likelihood, include_fixed=False)
            for p_name in lik_params.keys():
                self.model[p_name].fix()
            try:
                self.model.optimize(max_iters=max_iter, messages=messages)
            except Exception as e:
                logging.warning('optimization failed with exception ' + str(e))

            # optimize kernel and likelihood parameters
            for p_name in lik_params.keys():
                self.model[p_name].unfix()
            self.model.optimize_restarts(optimizer="bfgs",
                                         num_restarts=num_restarts,
                                         max_iters=max_iter,
                                         robust=True,
                                         parallel=False,
                                         messages=messages)

    def predict(self, X):
        """
        predict
        :param X: array (shape=[n_samples, n_dims], float). Features for prediction.
        :return: array (shape=[n_sample,_], int\float\str). predicted targets.
        """
        # get probabilities
        y_probs, y_labels_encoded = self.model.predict_proba(X)

        # decision based on arg-max
        y_pred_encoded = y_labels_encoded[np.argmax(y_probs, axis=1)]

        # decode
        y_labels = self.decode_targets(y_labels_encoded)
        y_pred = self.decode_targets(y_pred_encoded)

        return y_pred, y_probs, y_labels

    def predict_posterior_gp(self, X):
        """
        compute the mean and variance of the posterior predictive of the gp given input samples X, p(f*|X*, D).
        :param X:  array=[n_samples,n_dim], float. Input samples
        :return: tuple (mean, var). where mean is array (shape=[n_samples,1], float) with the mean values of p(f*|X*, D).
         var is array (shape=[n_samples,1], float) with the variance of p(f*|X*, D)
        """
        return self.model.predict_gp_posterior_stats(X)

    def sample_posterior_gp(self, X, n_samples):
        """
        sample from the posterior predictive p(f*|X*, D)
        :param X: array=[n_samples,n_dim], float. Input samples
        :param n_samples: int, number of samples to draw from the posterior \
        :return: array (shape=[n_samples,_], float). posterior samples
        """
        f = self.model.posterior_samples_f(X, size=n_samples).squeeze()
        return f

    def compute_log_marginal(self):
        """
        return the negative log of the marginal likelihood
        :param model: GPy model object
        :return: float, negative log of marginal likelihood
        """
        return self.model.log_likelihood()

    def compute_bic_cost(self):
        """
        compute bayesian information criterion (BIC). (we use BIC-cost definition given in Murphy 5.33. small is better)
        :return: float, bic criterion.
        """
        dof = np.size(self.model.parameter_names_flat(include_fixed=False))  # degree of freedom (no. params)
        log_marginal = self.compute_log_marginal()
        n_samples = np.size(self.model.Y, axis=0)
        bic_cost = -2*log_marginal + dof*np.log(n_samples)
        return bic_cost

    def compute_lppd(self):
        """
        compute the log pointwise predictive density (Bayesian Data Analysis, Ch. 7, Eq. 7.4). Also called Bayesian
        training utility. The lppd is overoptimisitic measure of predictive performance and therefore other performance
        metrics (e.g. waic) apply bias correction over the lppd.
        :return: float, Bayesian training utility
        """
        X_t = self.model.X  # input points (train set)
        Y_t = self.model.Y  # observations (train set)
        lpd = self.model.log_predictive_density(X_t, Y_t)
        bayes_train_util = np.mean(lpd)
        return bayes_train_util

    def compute_waic(self):
        """
        Compute WAIC metric
        :return: float, waic
        """
        # compute lppd
        btu = self.compute_lppd()

        # apply bias correction
        X_t = self.model.X  # input points (train set)
        Y_t = self.model.Y  # observations (train set)
        func_var = self.model.compute_var_log_y_cond_f(X_t, Y_t)
        bias_corr = np.mean(func_var)
        waic = btu - bias_corr
        return waic

    @staticmethod
    def get_obj_params(obj, include_fixed=True):
        """
        get likelihood params
        :param include_fixed: bool, if True return only non-fixed parameters.
        :return: dict, parameter value (value) per parameter name (key)
        """
        params_names = list(obj.parameter_names(add_self=True))
        params_dict = dict()
        for name in params_names:
            param = obj[name]
            if not include_fixed and param.is_fixed:
                continue
            params_dict.setdefault(name, param.values)
        return params_dict
