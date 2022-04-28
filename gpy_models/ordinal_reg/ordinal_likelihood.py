import numpy as np
from scipy import stats
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.core.parameterization import Param
from GPy import priors
from paramz.transformations import Logexp


class Ordinal(Likelihood):
    """
    A likelihood for doing ordinal regression.
    The labels are integer values from 0 to K. Additionaly (K-1) threshold values (hyper-parameters)
    are set to define the points at which the labels switch. Let the threshold
    be [a_0, a_1, ... a_{K-1}], then the likelihood is
    p(Y=0|F) = phi((a_0 - F) / sigma)
    p(Y=1|F) = phi((a_1 - F) / sigma) - phi((a_0 - F) / sigma)
    p(Y=2|F) = phi((a_2 - F) / sigma) - phi((a_1 - F) / sigma)
    ...
    p(Y=K|F) = 1 - phi((a_{K-1} - F) / sigma)
    where phi is the cumulative density function of a Gaussian (the inverse probit
    function) and sigma is a parameter to be learned. A reference is:
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
    def __init__(self,
                 eta1=0,
                 delta_vals=None,
                 delta_vals_prior_mean = 0,
                 delta_vals_prior_sigma = 0.25,
                 sigma=1,
                 gp_link=None,
                 fix_thresholds=False):
        """
        init Ordinal Likelihood.
        :param eta1: float, initial value for eta1 parameter.
        :param delta_vals: array (shape=[K-2,_], float). initial values for the delta parameters.
        :param delta_vals_prior_mean: float, mean value for logGaussian prior for delata parameters.
        :param delta_vals_prior_sigma: float, std value for logGaussian prior for delata parameters.
        :param sigma: float, noise variance
        :param gp_link: link function which is not used.
        :param fix_thresholds: bool, if true fix thresholds params.
        """
        super(Ordinal, self).__init__(link_functions.Identity(), name='Ordinal')
        # set sigma parameter (noise std)
        self.sigma = Param('Gaussian_noise', np.array([sigma]), Logexp())
        self.link_parameter(self.sigma)

        # set eta1 parameter (1st bin edge)
        self.eta1 = Param('eta1', np.array([eta1]))
        self.link_parameter(self.eta1)

        # set delta_vals parameters (padding)
        # assaf - NOTE: if_"delta_vals" is None, the ordinal regression likelihood reduces to binary (probit) likelihood
        if delta_vals is None or delta_vals.size == 0:
            self.delta_vals = None
            n_delta_vals = 0
        else:
            self.delta_vals = Param('delta_vals', delta_vals)
            self.delta_vals.set_prior(priors.LogGaussian(delta_vals_prior_mean,
                                                         delta_vals_prior_sigma))
            n_delta_vals = delta_vals.size
            self.link_parameter(self.delta_vals)

        if fix_thresholds:
            self.eta1.fix()
            if self.delta_vals is not None:
                self.delta_vals.fix()

        # set bin edges
        self.K = n_delta_vals + 2

        # assaf - NOTE: likelihood is concave function of the latent variable f.
        self.log_concave = True

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.
        Note: It uses the private method _save_to_input_dict of the parent.
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = dict()
        input_dict["class"] = "GPy.likelihoods.Ordinal"
        input_dict["Gaussian_noise"] = self.sigma.values.tolist()
        input_dict["eta1"] = self.eta1.values.tolist()
        if self.delta_vals is not None:
            input_dict['delta_vals'] = self.delta_vals.values.tolist()

        return input_dict

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        self.sigma.gradient = grads[0]
        self.eta1.gradient = grads[1]
        if self.delta_vals is not None:
            self.delta_vals.gradient = grads[2:]

    def get_bin_edges(self):
        """
        compute the bin edges.
        :return: array (shape=[K-1,_], float), bin edges.
        """
        if self.delta_vals is None:
            bin_edges = self.eta1
        else:
            bin_edges = self.eta1 + np.cumsum(np.append(0, self.delta_vals))
        return bin_edges

    def get_z_arg(self, f, y):
        """
        Auxiliary function to get the z1_i and z2_i arguments (Eq. 25)
        :param f: array (shape=[N,_], float). Latent variable f.
        :param y: array (shape=[N,_], int). Targets
        :return: tuple (z1_i, z2_i), each is array (shape=[N,_], float).
        """
        y = y.astype(int)
        bin_edges = self.get_bin_edges()
        scaled_bins_left = np.append(bin_edges / (self.sigma + np.finfo(float).eps), np.array([np.inf]))
        scaled_bins_right = np.append(np.array([-np.inf]), bin_edges / (self.sigma + np.finfo(float).eps))
        selected_bins_left = scaled_bins_left[y]
        selected_bins_right = scaled_bins_right[y]
        z1_i = selected_bins_left - f / (self.sigma + np.finfo(float).eps)
        z2_i = selected_bins_right - f / (self.sigma + np.finfo(float).eps)
        return z1_i, z2_i

    def compute_v_p(self, f, y, p):
        """
        Auxiliary function to compute v_p (Eq. 38)
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets
        :param p: int.
        :return: float, v_p as in Eq. (38)
        """
        z1_i, z2_i = self.get_z_arg(f, y)
        z1_i_p = z1_i**p
        z2_i_p = z2_i**p
        # assaf - NOTE: during optimization we might get infinite values for f which result in infinite values
        # for z1_i and z2_i. since inf*stats.norm.pdf(inf) = 0. We set z1_i and z2_i to zero in this case.
        z1_i_p[np.isinf(z1_i_p)] = 0
        z2_i_p[np.isinf(z2_i_p)] = 0
        num = z1_i_p*stats.norm.pdf(z1_i) - z2_i_p*stats.norm.pdf(z2_i)
        den = stats.norm.cdf(z1_i) - stats.norm.cdf(z2_i)
        v_p = num/(den + np.finfo(float).eps)
        return v_p

    def compute_s_p(self, f, y, p):
        """
        Auxiliary function to compute s_p (Eq. 55)
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets
        :param p: int.
        :return: float, v_p as in Eq. (38)
        """
        z1_i, z2_i = self.get_z_arg(f, y)
        z1_i_p = z1_i**p
        # assaf - NOTE: during optimization we might get infinite values for f which result in infinite values
        # for z1_i. since inf*stats.norm.pdf(inf) = 0. We set z1_i to zero in this case.
        z1_i_p[np.isinf(z1_i_p)] = 0
        num = z1_i_p*stats.norm.pdf(z1_i)
        den = stats.norm.cdf(z1_i) - stats.norm.cdf(z2_i)
        s_p = num/(den + np.finfo(float).eps)
        return s_p

    def pdf_cond_yi(self, f, y):
        """
        compute the conditional distribution of every point y_i given f_i. (Eq. 4).
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets
        :return: array (shape=[N,1], float). conditional distribution p(yi|fi) evaluated at given points.
        """
        z1_i, z2_i = self.get_z_arg(f, y)
        p_yi_fi = stats.norm.cdf(z1_i) - stats.norm.cdf(z2_i)
        return p_yi_fi

    def pdf_link(self, f, y, Y_metadata=None):
        """
        Likelihood function given f
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets
        :param Y_metadata: Y_metadata which is not used
        :return: float, likelihood of the data given f
        """
        p_yi_fi = self.pdf_cond_yi(f, y)
        return np.prod(p_yi_fi)

    def logpdf_link(self, f, y, Y_metadata=None):
        """
        log-Likelihood function given f
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets
        :param Y_metadata: Y_metadata which is not used
        :return: float, log-likelihood of the data given f
        """
        p_yi_fi = self.pdf_cond_yi(f, y)
        return np.sum(np.log(p_yi_fi + np.finfo(float).eps))

    def dlogpdf_dlink(self, f, y, Y_metadata=None):
        """
        gradient of the log likelihood at y given f, w.r.t f (Eq. 26)
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets
        :param Y_metadata: Y_metadata which is not used
        :return: array (shape=[N,1], float) gradient of likelihood evaluated at points y.
        """
        v0 = self.compute_v_p(f, y, p=0)
        dp = -(1/self.sigma)*v0
        return dp

    def d2logpdf_dlink2(self, f, y, Y_metadata=None):
        """
        Second order derivative of log-likelihood function at y given f w.r.t f (Eq. 29).
        :param f: array (shape=[N, 1], float). Latent variable f.
        :param y: array (shape=[N, 1], int). Targets
        :param Y_metadata: Y_metadata which is not used
        :return: array (shape=[N, 1], float). Diagonal of Hessian matrix.
        """
        v0 = self.compute_v_p(f, y, p=0)
        v1 = self.compute_v_p(f, y, p=1)
        d2p = -(v1 + v0**2)/(self.sigma**2 + np.finfo(float).eps)
        # assaf - NOTE: elements are non-positive (hessian is positive definite)
        d2p[d2p > -np.finfo(float).eps] = -np.finfo(float).eps
        return d2p

    def d3logpdf_dlink3(self, f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given f w.r.t f (Eq. )
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used
        :return: array (shape=[N,1], float). Third order derivative.
        """
        v0 = self.compute_v_p(f, y, p=0)
        v1 = self.compute_v_p(f, y, p=1)
        v2 = self.compute_v_p(f, y, p=2)
        d3p = -(2*v0**3 + 3*v0*v1 + v2 - v0)/(self.sigma**3 + np.finfo(float).eps)
        return d3p

    def dlogpdf_link_dsigma(self, f, y, Y_metadata=None):
        """
        gradient of the log-likelihood w.r.t sigma parameter
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[N,1], float).
        """
        v1 = self.compute_v_p(f, y, p=1)
        dl_dsigma = -(1/self.sigma)*v1
        return dl_dsigma

    def dlogpdf_dlink_dsigma(self, f, y, Y_metadata=None):
        """
        gradient of the log-likelihood w.r.t sigma parameter
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[N,1], float).
        """
        v0 = self.compute_v_p(f, y, p=0)
        v1 = self.compute_v_p(f, y, p=1)
        v2 = self.compute_v_p(f, y, p=2)
        dl_df_dsigma = -(v2 + v0*v1)/(self.sigma**2 + np.finfo(float).eps)
        return dl_df_dsigma

    def d2logpdf_dlink2_dsigma(self, f, y, Y_metadata=None):
        """
        gradient of d2logpdf_dlink2 w.r.t. sigma parameter
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[N,1], float).
        """
        v0 = self.compute_v_p(f, y, p=0)
        v0_2 = v0**2
        v1 = self.compute_v_p(f, y, p=1)
        v1_2 = v1**2
        v2 = self.compute_v_p(f, y, p=2)
        v3 = self.compute_v_p(f, y, p=3)
        d2l_df2_dsigma = -(2*v0*v2 + 2*v0_2*v1 - v1 + v1_2 + v3)/(self.sigma**3 + np.finfo(float).eps)
        return d2l_df2_dsigma

    def dlogpdf_link_deta1(self, f, y, Y_metadata=None):
        """
        gradient of the log-likelihood w.r.t eta parameter
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[N,1], float).
        """
        v0 = self.compute_v_p(f, y, p=0)
        dl_deta1 = (1/self.sigma)*v0
        return dl_deta1

    def dlogpdf_dlink_deta1(self, f, y, Y_metadata=None):
        """
        gradient of the log-likelihood w.r.t eta1 parameter
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[N,1], float).
        """
        v0 = self.compute_v_p(f, y, p=0)
        v0_2 = v0**2
        v1 = self.compute_v_p(f, y, p=1)
        dl_df_deta1 = (v0_2 + v1)/(self.sigma**2 + np.finfo(float).eps)
        return dl_df_deta1

    def d2logpdf_dlink2_deta1(self, f, y, Y_metadata=None):
        """
        gradient of d2logpdf_dlink2 w.r.t. eta1 parameter
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[N,1], float).
        """
        v0 = self.compute_v_p(f, y, p=0)
        v0_3 = v0**3
        v1 = self.compute_v_p(f, y, p=1)
        v2 = self.compute_v_p(f, y, p=2)
        d2l_df2_dsigma = (2*v0_3 + 3*v0*v1 - v0 + v2)/(self.sigma**3 + np.finfo(float).eps)
        return d2l_df2_dsigma

    def dlogpdf_link_ddelta(self, f, y, Y_metadata=None):
        """
        gradient of the log-likelihood w.r.t delta vals parametera
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[K-2,N,1], float), column k give the derivative w.r.t delta_vals[k].
        """
        # init
        n_delta = self.K - 2
        if n_delta == 0:
            raise ValueError("Model is binary, no delta parameters.")
        dl_ddelta = np.zeros([n_delta, f.shape[0], 1])

        # compute derivatives
        s0 = self.compute_s_p(f, y, p=0)
        v0 = self.compute_v_p(f, y, p=0)
        d_eq_l = (1/self.sigma)*s0
        d_gt_l = (1/self.sigma)*v0

        # set conditions
        for l in range(n_delta):
            arg_eq_l = np.nonzero(y-1 == l)[0]
            arg_gt_l = np.nonzero(y-1 > l)[0]
            dl_ddelta[l, arg_eq_l, 0] = d_eq_l[arg_eq_l].flatten()
            dl_ddelta[l, arg_gt_l, 0] = d_gt_l[arg_gt_l].flatten()
        return dl_ddelta

    def dlogpdf_dlink_ddelta(self, f, y, Y_metadata=None):
        """
        gradient of the log-likelihood w.r.t delta vals parameters
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[K-2,N,1], float).
        """
        # init
        n_delta = self.K - 2
        if n_delta == 0:
            raise ValueError("Model is binary, no delta parameters.")
        dl_df_ddelta = np.zeros([n_delta, f.shape[0], 1])

        # compute derivatives
        v0 = self.compute_v_p(f, y, p=0)
        v0_2 = v0**2
        v1 = self.compute_v_p(f, y, p=1)
        s0 = self.compute_s_p(f, y, p=0)
        s1 = self.compute_s_p(f, y, p=1)
        d_eq_l = (v0*s0 + s1)/(self.sigma**2 + np.finfo(float).eps)
        d_gt_l = (v0_2 + v1)/(self.sigma**2 + np.finfo(float).eps)

        # set conditions
        for l in range(n_delta):
            arg_eq_l = np.nonzero(y-1 == l)[0]
            arg_gt_l = np.nonzero(y-1 > l)[0]
            dl_df_ddelta[l, arg_eq_l, 0] = d_eq_l[arg_eq_l].flatten()
            dl_df_ddelta[l, arg_gt_l, 0] = d_gt_l[arg_gt_l].flatten()
        return dl_df_ddelta

    def d2logpdf_dlink2_ddelta(self, f, y, Y_metadata=None):
        """
        gradient of d2logpdf_dlink2 w.r.t. delta vals parametera
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[K-2,N,1], float).
        """
        # init
        n_delta = self.K - 2
        if n_delta == 0:
            raise ValueError("Model is binary, no delta parameters.")
        d2l_df2_ddelta = np.zeros([n_delta, f.shape[0], 1])

        # compute derivatives
        v0 = self.compute_v_p(f, y, p=0)
        v0_2 = v0**2
        v0_3 = v0**3
        v1 = self.compute_v_p(f, y, p=1)
        v2 = self.compute_v_p(f, y, p=2)
        s0 = self.compute_s_p(f, y, p=0)
        s1 = self.compute_s_p(f, y, p=1)
        s2 = self.compute_s_p(f, y, p=2)
        d_eq_l = (2*v0_2*s0 + 2*v0*s1 - s0 + s2 + v1*s0)/(self.sigma**3 + np.finfo(float).eps)
        d_gt_l = (2*v0_3 + 3*v0*v1 - v0 + v2)/(self.sigma**3 + np.finfo(float).eps)

        # set conditions
        for l in range(n_delta):
            arg_eq_l = np.nonzero(y-1 == l)
            arg_gt_l = np.nonzero(y-1 > l)
            d2l_df2_ddelta[l, arg_eq_l, 0] = d_eq_l[arg_eq_l].flatten()
            d2l_df2_ddelta[l, arg_gt_l, 0] = d_gt_l[arg_gt_l].flatten()
        return d2l_df2_ddelta

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        """
        gradient of the likelihood w.r.t to theta
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[n_params,N, 1], float). gradient values.
        """
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0, :, :] = self.dlogpdf_link_dsigma(f, y)
        dlogpdf_dtheta[1, :, :] = self.dlogpdf_link_deta1(f, y)
        if self.delta_vals is not None:
            dlogpdf_dtheta[2:, :, :] = self.dlogpdf_link_ddelta(f, y)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        """
        gradient of dlogpdf_dlink w.r.t to theta. Where dlogpdf_dlink is the gradient of the likelihood w.r.t f.
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[n_params,N, 1], float). gradient values.
        """
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0, :, :] = self.dlogpdf_dlink_dsigma(f, y, Y_metadata=Y_metadata)
        dlogpdf_dlink_dtheta[1, :, :] = self.dlogpdf_dlink_deta1(f, y, Y_metadata=Y_metadata)
        if self.delta_vals is not None:
            dlogpdf_dlink_dtheta[2:, :, :] = self.dlogpdf_dlink_ddelta(f, y)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        """
        gradient of d2logpdf_dlink2 w.r.t to theta. Where d2logpdf_dlink2 is the hessian of the likelihood w.r.t f.
        :param f: array (shape=[N,1], float). Latent variable f.
        :param y: array (shape=[N,1], int). Targets.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[n_params,N, 1], float). gradient values.
        """
        d2logpdf_dlink2_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink2_dtheta[0, :, :] = self.d2logpdf_dlink2_dsigma(f, y, Y_metadata=Y_metadata)
        d2logpdf_dlink2_dtheta[1, :, :] = self.d2logpdf_dlink2_deta1(f, y, Y_metadata=Y_metadata)
        if self.delta_vals is not None:
            d2logpdf_dlink2_dtheta[2:, :, :] = self.d2logpdf_dlink2_ddelta(f, y)
        return d2logpdf_dlink2_dtheta

    def compute_prob_mat(self, f):
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.
        Note that a matrix of F values is flattened.
        :param f: array (shape=[N,1], float). Latent variable f.
        :return: array (shape=[N, K], float). matrix, the ij element is p(y==j|fi)
        """
        bin_edges = self.get_bin_edges()
        scaled_bins_left = np.append(bin_edges / self.sigma, np.array([np.inf]))
        scaled_bins_right = np.append(np.array([-np.inf]), bin_edges / self.sigma)
        return stats.norm.cdf(scaled_bins_left - f/self.sigma) - stats.norm.cdf(scaled_bins_right - f/self.sigma)

    def samples(self, gp, Y_metadata=None, samples=1, seed=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.
        :param gp: array (shape = [N, M]. latent variable
        :param Y_metadata: Y_metadata which is not used.
        :param samples: int, number of samples to generate from each p(y|fi).
        :return array (shape = [N,samples], int). samples os Ys.
        """
        Ys = np.arange(self.K)
        Ps = self.compute_prob_mat(gp)
        samples = [stats.rv_discrete(values=(Ys, Ps[i, :])).rvs(size=samples, random_state=seed) for i in range(gp.shape[0])]
        return np.array(samples)

    def conditional_mean(self, gp):
        """
        The mean of the random variable y conditioned on one value of the GP (f)
        """
        Ys = np.arange(self.K).reshape(-1, 1)
        Ps = self.compute_prob_mat(gp)
        E_y = np.matmul(Ps, Ys)
        return E_y

    def conditional_variance(self, gp):
        """
        The variance of the random variable y conditioned on one value of the GP (f)
        """
        Ys = np.arange(self.K).reshape(-1, 1)
        Ps = self.compute_prob_mat(gp)
        E_y = np.matmul(Ps, Ys)
        E_y2 = np.matmul(Ps, Ys**2)
        return E_y2 - E_y**2

    def posterior_predictive_y(self, mu_star, var_star, Y_metadata=None):
        """
        compute the posterior predictive distribution (Eq. 23).
        :param mu_star: array(shape=[n_sample,_], float ). mean values of the posterior predictive of f.
        :param var_star: array(shape=[n_sample,_], float ). variance values of the posterior predictive of f.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[n_sample,K], float). probability per each possible value of Y.
        """
        bin_edges = self.get_bin_edges()
        scale_factor = np.sqrt(var_star + self.sigma**2)
        selected_bins_left = np.append(bin_edges, np.array([np.inf]))
        selected_bins_right = np.append(np.array([-np.inf]), bin_edges)
        arg_left = (selected_bins_left - mu_star) / scale_factor
        arg_right = (selected_bins_right - mu_star) / scale_factor
        p = stats.norm.cdf(arg_left) - stats.norm.cdf(arg_right)
        return p

    def log_predictive_density(self, y_test, mu_star, var_star, Y_metadata=None):
        """
        compute the log (posterior) predictive distribution p(yi|xi,D)
        :param y_test: array(shape=[n_sample,1], int), query points.
        :param mu_star: array(shape=[n_sample,_], float ). mean values of the posterior predictive of f.
        :param var_star: array(shape=[n_sample,_], float ). variance values of the posterior predictive of f.
        :param Y_metadata: not used.
        :return: array (shape=[n_sample,1], float). predictive distribution p(yi|xi,D)
        """
        predictive_prob_mat = self.posterior_predictive_y(mu_star, var_star)
        y_ind = y_test.squeeze().astype(int)
        return np.log(predictive_prob_mat[range(len(y_ind)), y_ind])

    def get_nb_ordinal_vars(self):
        """
        get the number of ordinal variables.
        :return: int, number of ordinal variables
        """
        return self.K

