import numpy as np
from GPy.core import GP
from GPy import kern
from GPy.inference.latent_function_inference import Laplace
from gpy_models.ordinal_reg.ordinal_likelihood import Ordinal


class GPOrdinalRegression(GP):
    """
    Gaussian Process Ordinal Regression Model.

    A reference is:
    @article{chu2005gaussian,
      title={Gaussian processes for ordinal regression},
      author={Chu, Wei and Ghahramani, Zoubin},
      journal={Journal of Machine Learning Research},
      volume={6},
      number={Jul},
      pages={1019--1041},
      year={2005}
    }
    """
    def __init__(self, X, Y, kernel=None, mean_function=None, inference_method=None, likelihood=None):
        """

        :param X: array (shape = [N,D], float). Input vector (features).
        :param Y: array (shape = [N, 1], int). observed ordinal variables.
        :param kernel: a GPy kernel, defaults to linear.
        :param inference_method: Latent function inference to use, defaults to Laplace.
        :param likelihood: a GPy likelihood.
        """
        # cfg kernel
        if kernel is None:
            kernel = kern.Linear(input_dim=X.shape[1])

        # cfg likelihood
        if likelihood is None:
            likelihood = Ordinal()

        # cfg inference method
        if inference_method is None:
            inference_method = Laplace()

        # input validation
        k = likelihood.get_nb_ordinal_vars()
        if np.max(Y) > k-1 or np.min(Y) < 0 or not np.equal(np.mod(Y, 1), 0).all():
            raise ValueError("Y must be integer in the range [0,{:d}]".format(k))

        # init
        GP.__init__(self, X=X, Y=Y,  kernel=kernel, likelihood=likelihood, inference_method=inference_method,
                    mean_function=mean_function, name='GPOrdinalRegression', normalizer=False)

    def predict_gp_posterior_stats(self, Xnew):
        """
        compute the mean and variance of the posterior predictive of the gp given input samples X, p(f*|X*, D).
        :param Xnew:  array=[n_samples,n_dim], float. Input samples
        :return: tuple (mean, var). where mean is array (shape=[n_samples,1], float) with the mean values of p(f*|X*, D).
         var is array (shape=[n_samples,1], float) with the variance of p(f*|X*, D)
        """
        return self.predict_noiseless(Xnew)

    def predict_proba(self, Xnew):
        """
        Predict the probabilities at the new point(s) Xnew.
        :param Xnew: (array=[n_samples,n_dim], float). The points at which to make a prediction
        :return tuple (prob, y_labels), prob is array (shape=[n_samples, K], int) where element i,j equals p[y==j|x_i].
         y_labels is array (shape = [n_labels,_] with labels.
        """
        # posterior predictive for the latent variable
        mean, var = self.predict_gp_posterior_stats(Xnew)

        # posterior predictive for y
        prob = self.likelihood.posterior_predictive_y(mean, var)

        y_labels = np.arange(self.likelihood.get_nb_ordinal_vars())
        return prob, y_labels

    def compute_entopy_y(self, X):
        """
        compute the entropy of y_i using the the posterior predictive distribution p(y_i|x_i,D).
        :param X: (array=[n_samples,n_dim], float). Input samples
        :return: (array=[n_samples,_], float). Entropy of y_i.
        """
        # compute the posterior predictive p(y_i|x_i,D)
        prob, _ = self.predict_proba(X)

        # compute entropy
        p_log_p = prob*np.log(prob, out=np.zeros_like(prob), where=prob != 0)
        entropy = -np.sum(p_log_p, axis=1)
        return entropy

    def compute_entropy_y_cond_f(self, f):
        """
        compute the entropy of y given the latent f, sum_j{p(y_j|f_i)*log[p(y_j|f_i)}.
        :param f: array (shape=[N,1], float).  gp values.
        :return: array (shape=[N,1], float). Entropy values.
        """
        # compute  p(y=j|f_i, X_i). This results in a matrix of size [N, K]
        p_y_f = self.likelihood.compute_prob_mat(f)

        # compute entropies
        p_log_p = p_y_f*np.log(p_y_f, out=np.zeros_like(p_y_f), where=p_y_f != 0)
        entropy = -np.sum(p_log_p, axis=1)
        return entropy

    def compute_exected_entropy_y_cond_f(self, X, method='riemann_sum', n_samples=5000):
        """
        compute the expectation of the entropy of y given the latent f (GP). The expectation is taken w.r.t to the
        latent GP f and is computed using numerical integration.
        :param X: (array=[Nx,n_dim], float). Input samples
        :param method: string, method used to approximate the expectation. Support values are 'sampling' and
         'riemann_sum'.
        :param n_samples: number of sampling points.
        :return: array (shape=[Nx,_], float). Expected entropy.
        """
        # init
        expetced_entropy = 0

        if method == 'sampling':
            # sample from the posterior predictive of f
            f = self.posterior_samples_f(X, size=n_samples).squeeze()  # [Nx, n_samples]
            # approximate the expectation by sum (SLLN)
            expetced_entropy = 0
            for n in range(n_samples):  # loop over f samples
                f_n = f[:, n].reshape(-1, 1)
                expetced_entropy += (1 / n_samples) * self.compute_entropy_y_cond_f(f_n)
        elif method == 'riemann_sum':
            # compute mean and variance of the posterior
            mean_f, var_f = self.predict_gp_posterior_stats(X)
            # approximate the expectation by Riemann sum
            std = 3
            z, delta_z = np.linspace(-std, std, n_samples, retstep=True)
            for n in range(n_samples):
                arg = np.sqrt(var_f)*z[n] + mean_f  # change of variables
                pdf_z = (1 / np.sqrt(2*np.pi))*np.exp(-0.5*z[n]**2)
                expetced_entropy += delta_z*pdf_z*self.compute_entropy_y_cond_f(arg)
        else:
            raise ValueError("unsupported method")
        return expetced_entropy

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        """
        compute the log predictive density log[p(y_i|x_i,D)]
        :param x_test: array (shape = [N,D], float). Input points.
        :param y_test: array (shape=[N,1], int). Query points.
        :param Y_metadata: not used.
        :return: array (shape=[N,1], float). Log predictive density values.
        """
        # posterior predictive for the latent variable
        mu_star, var_star = self.predict_gp_posterior_stats(x_test)

        # log posterior predictive log[p(y_i|x_i,D)]
        lpd = self.likelihood.log_predictive_density(y_test, mu_star, var_star)

        return lpd

    def compute_var_log_y_cond_f(self, x_test, y_test, method='riemann_sum', num_samples=10000):
        """
        compute the variance of log{p[y_i|f(x_i)]} w.r.t the posterior p(f(x_i)|D).
        :param x_test: array (shape = [N,D], float). Input points.
        :param y_test: array (shape=[N,1], int). Query points.
        :param method: string, method used to approximate the expectation. Support values are 'sampling' and
         'riemann_sum'.
        :param num_samples: int, number of posterior sample to take.
        :return: array (shape=[N,1], float). variance of log predictive density.
        """
        # init
        E_logp = 0
        E_logp_square = 0
        eps = 1e-3
        # posterior predictive for the latent variable
        mu_star, var_star = self.predict_gp_posterior_stats(x_test)

        # approximate the expectations by Riemann sum
        if method == 'sampling':
            f_samples = np.random.normal(mu_star, np.sqrt(var_star), size=(mu_star.shape[0], num_samples))
            for n in range(num_samples):
                f_n = f_samples[:, n].reshape(-1, 1)
                p_cond_y = self.likelihood.pdf_cond_yi(f_n, y_test)
                log_pdf = np.log(p_cond_y, where=p_cond_y > eps, out=np.ones_like(p_cond_y)*np.log(eps))
                E_logp += (1 / num_samples) * log_pdf
                E_logp_square += (1 / num_samples) * log_pdf**2
        elif method == 'riemann_sum':
            std = 3
            z, delta_z = np.linspace(-std, std, num_samples, retstep=True)
            for n in range(num_samples):
                arg = np.sqrt(var_star) * z[n] + mu_star  # change of variables
                pdf_z = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z[n] ** 2)
                p_cond_y = self.likelihood.pdf_cond_yi(arg, y_test)
                log_pdf = np.log(p_cond_y, where=p_cond_y > eps, out=np.ones_like(p_cond_y)*np.log(eps))
                E_logp += delta_z * pdf_z * log_pdf
                E_logp_square += delta_z * pdf_z * log_pdf**2
        else:
            raise ValueError("unsupported method")
        var_logp = E_logp_square - E_logp**2
        return var_logp

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        GPOrdinalRegression(gp.X, gp.Y, gp.kern, gp.mean_function, gp.inference_method, gp.likelihood)

    def to_dict(self, save_data=True):
        model_dict = super(GPOrdinalRegression, self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPOrdinalRegression"
        return model_dict

    @staticmethod
    def from_dict(input_dict, data=None):
        import GPy
        m = GPy.core.model.Model.from_dict(input_dict, data)
        return GPOrdinalRegression.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)

    @staticmethod
    def _build_from_input_dict(input_dict, data=None):
        input_dict = GPOrdinalRegression._format_input_dict(input_dict, data)
        input_dict.pop('name', None)  # Name parameter not required by GPOrdinalRegression
        return GPOrdinalRegression(**input_dict)
