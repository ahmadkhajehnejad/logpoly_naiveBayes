from .tools import mp_compute_poly, mp_log_integral_exp, mp_compute_SS, mp_moments # , mp_integral
import config.logpoly
import config.general
import numpy as np
from tools import get_train_and_validation_index
from mpmath import mp, mpf
import mpmath
import warnings
import scipy.integrate as integrate
# import os
# import matplotlib.pyplot as plt
import sys
import logpoly

class Logpoly:

    def __init__(self, min_unscaled_value, max_unscaled_value):
        self.min_unscaled_value = min_unscaled_value
        self.max_unscaled_value = max_unscaled_value


    def fit(self, SS, n, constant_bias = 1, plot=False):

        k = SS.size - 1

        self.theta = np.array([mpf(0) for i in range(k+1)])
        if constant_bias is None:
            self.theta[0] = mpf(1)
        else:
            self.theta[0] = mpf(constant_bias)
        self.current_log_likelihood = None

        for iteration in range(config.logpoly.Newton_max_iter):

            if config.logpoly.verbose:
                print('.')
                print('                                 iteration #', iteration)
                sys.stdout.flush()

            if iteration == 0:
                self.current_log_likelihood, self.logZ, roots = _compute_log_likelihood(SS, self.theta, n)

            ## Compute sufficient statistics and constructing the gradient and the Hessian

            if constant_bias is None:
                grad_dimensions = np.arange(k+1)
            else:
                grad_dimensions = np.arange(1, k+1)

            #######################

            if config.logpoly.verbose:
                print('compute_Expectations start')
                sys.stdout.flush()

            def func(x):
                return (mpmath.exp(1) * np.ones(x.shape)) ** (mp_compute_poly(x, self.theta) - self.logZ)

            tmp = mp_moments(func, 2*k+1, np.unique(np.concatenate([np.array([mpf(0)]), roots])),
                             config.logpoly.x_lbound, config.logpoly.x_ubound)
            ESS = mpmath.matrix(tmp[grad_dimensions])

            H = mpmath.matrix(len(grad_dimensions))
            for i in range(len(grad_dimensions)):
                for j in range(len(grad_dimensions)):
                    H[i, j] = tmp[grad_dimensions[i]+grad_dimensions[j]]

            H = -n*( H - (ESS * ESS.transpose()) )
            grad = mpmath.matrix(SS[grad_dimensions]) - n*ESS

            if config.logpoly.verbose:
                print('compute_Expectations finish')
                sys.stdout.flush()

            if config.logpoly.verbose:
                print('solve_inversion start')
                sys.stdout.flush()

            delta_theta_subset = mp.lu_solve(H, -grad)
            lambda2 = (grad.transpose() * delta_theta_subset)[0]
            delta_theta_subset = np.array(delta_theta_subset)
            delta_theta = np.array([mpf(0) for _ in range(k+1)])
            delta_theta[grad_dimensions] = delta_theta_subset

            if config.logpoly.verbose:
                print('solve_inversion finish')
                sys.stdout.flush()

            if config.logpoly.verbose:
                print('current_log_likelihood = ', self.current_log_likelihood)
                print('lambda_2 / 2 = ', lambda2/2)
                sys.stdout.flush()

            if lambda2 < 0:
                warnings.warn('lambda_2 < 0')
            if lambda2 / 2 < n*config.logpoly.theta_epsilon:
                if config.logpoly.verbose:
                    print('%')
                    sys.stdout.flush()
                break

            ## Line search
            lam = 1
            alpha = 0.49; beta = 0.5

            while True:
                tmp_log_likelihood, tmp_logZ, tmp_roots = _compute_log_likelihood(SS, self.theta + lam * delta_theta, n)

                if tmp_log_likelihood < self.current_log_likelihood + alpha * lam * np.inner(grad, delta_theta_subset):
                    if config.logpoly.verbose:
                        print('+', end='')
                        sys.stdout.flush()
                    lam = lam * beta
                else:
                    break

            if tmp_log_likelihood <= self.current_log_likelihood:
                if config.logpoly.verbose:
                    print('*')
                    sys.stdout.flush()
                break

            self.theta = self.theta + lam * delta_theta
            self.current_log_likelihood = tmp_log_likelihood
            self.logZ = tmp_logZ
            roots = tmp_roots

            # print(self.theta)
            # sys.stdout.flush()
            # print(self.current_log_likelihood)
            # sys.stdout.flush()
            # if plot:
            #     plt.cla()
            #
            #     l = config.logpoly.x_ubound - config.logpoly.x_lbound
            #     plt.hist(self.interface.x,np.arange(config.logpoly.x_lbound + (l/200), config.logpoly.x_ubound, l/100), density=True)
            #
            #     x = np.arange(config.logpoly.x_lbound, config.logpoly.x_ubound, 0.001)
            #     p = np.exp(self.logpdf(x))
            #     plt.plot(x,p)
            #     plt.ylim([0,5])
            #     plt.text(0,4,'log likelihood:\n      ' + str(self.current_log_likelihood))
            #     plt.show()
            #     plt.savefig('./log/' + str(i) + '.png')
            #     plt.close()



    def logpdf(self, x):
        p = mp_compute_poly(x, self.theta)
        return np.array( [ (a - self.logZ) for a in p ], dtype=float)


class LogpolyModelSelector:
    def __init__(self, list_factor_degrees):
        self.list_factor_degrees = list_factor_degrees

    def select_model(self, data):

        if config.logpoly.single_scale:
            min_value = None
            max_value = None
            scaled_data = data
        else:
            min_value = np.min(data)
            max_value = np.max(data)
            scaled_data = logpoly.tools.scale_data(data, min_value, max_value)

        n_total = data.shape[0]

        if len(self.list_factor_degrees) == 1:
            SS = mp_compute_SS(scaled_data, self.list_factor_degrees[0])
            logpoly_model = Logpoly(min_value, max_value)
            logpoly_model.fit(SS,n_total)
            return logpoly_model

        if config.classifier.smart_validation:
            ind = np.argsort(scaled_data)
        else:
            ind = np.arange(n_total)
        index_train, index_validation = get_train_and_validation_index(ind)
        n_train = index_train.size

        avg_log_likelihoods = []
        logpoly_models = []
        SS = mp_compute_SS(scaled_data[index_train], np.max(self.list_factor_degrees))
        for i, k in enumerate(self.list_factor_degrees):
            logpoly_models.append(Logpoly(min_value, max_value))
            logpoly_models[i].fit(SS[:k+1], n_train)
            avg_log_likelihoods.append(np.mean(logpoly_models[i].logpdf(scaled_data[index_validation])))

        best_index = np.argmax(avg_log_likelihoods)
        k = self.list_factor_degrees[best_index]
        best_logpoly_model = Logpoly(min_value, max_value)
        best_logpoly_model.fit(SS[:k+1] + mp_compute_SS(scaled_data[index_validation], k), n_total)
        return best_logpoly_model


def _compute_log_likelihood(SS, theta, n):
    if config.logpoly.verbose:
        print('_compute_log_likelihood start')
        sys.stdout.flush()

    #logZ = log_integral_exp( compute_poly, theta, previous_critical_points)
    # ll = -n*logZ + np.inner(theta[-1,:].reshape([-1,]), SS.reshape([-1,]))
    if config.logpoly.verbose:
        print('    compute_logZ start')
        sys.stdout.flush()
    logZ, roots = mp_log_integral_exp( mp_compute_poly, theta)
    if config.logpoly.verbose:
        print('    compute_logZ finish')
        sys.stdout.flush()
    ll = -n * logZ + np.inner(theta, SS)
    if config.logpoly.verbose:
        print('_compute_log_likelihood finish')
        sys.stdout.flush()
    return ll, logZ, roots

