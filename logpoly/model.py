from .tools import mp_compute_poly, mp_log_integral_exp, mp_compute_SS #, mp_integral
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

class Logpoly:

    def __init__(self, factor_degree, factors_count=1):
        self.factor_degree = factor_degree
        self.factors_count = factors_count
        self.theta = np.array([])
        mp.dps = config.logpoly.mp_dps


    def _fit_new_factor(self, SS, n, constant_bias = None):
        theta = self.theta

        k = self.factor_degree

        theta_new = np.array([mpf(0) for i in range(k+1)])
        if constant_bias is None:
            theta_new[0] = mpf(1)
        else:
            theta_new[0] = mpf(constant_bias)
        current_log_likelihood = None
        
        if theta is None:
            theta = np.array([], dtype=float)
        
        for iteration in range(config.logpoly.Newton_max_iter):
            
            # print('.',end='')
            print('.')
            print('fit_new_facor start')
            sys.stdout.flush()

            ## Compute sufficient statistics and constructing the gradient and the Hessian
            ESS = np.array([mpf(0) for i in range(k+1)])
            print('compute_logZ start')
            sys.stdout.flush()
            logZ = mp_log_integral_exp(mp_compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]))
            print('compute_logZ finish')
            sys.stdout.flush()

            print('compute_Expectations start')
            sys.stdout.flush()
            for j in range(k+1):
                print(j, '- ', end='')
                sys.stdout.flush()

                if len(theta) > 0:
                    def func(x):
                        return mp_compute_poly(x, theta)[0] * mpmath.power(x,j) * mpmath.exp(mp_compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])]))[0] - logZ)
                else:
                    def func(x):
                        return mpmath.power(x,j) * mpmath.exp(mp_compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])]))[0] - logZ)
                    
                ESS[j] = mpmath.quad(func, [config.logpoly.x_lbound, config.logpoly.x_ubound])

            ESS = mpmath.matrix(ESS)

            tmp = np.array([mpf(0) for i in range(2*k+1)])
            for j in range(2*k+1):
                if len(theta) > 0:
                    def func(x):
                        return (mp_compute_poly(x, theta)[0] ** 2) * mpmath.power(x,j) * mpmath.exp(mp_compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])]))[0] - logZ)
                else:
                    def func(x):
                        return mpmath.power(x,j) * mpmath.exp(mp_compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])]))[0] - logZ)

                if (len(theta) == 0) and (j <= k):
                    tmp[j] = ESS[j]
                else:
                    print(j, '- ', end='')
                    sys.stdout.flush()
                    tmp[j] = mpmath.quad(func, [config.logpoly.x_lbound, config.logpoly.x_ubound])

            print()

            H = mpmath.matrix(k+1)
            for i in range(k+1):
                for j in range(k+1):
                    H[i, j] = tmp[i+j]

            H = -n*( H - (ESS * ESS.transpose()) )
            grad = mpmath.matrix(SS) - n*ESS

            # if constant_bias is None:
            #     pass
            # else:
            #     eps = 1e-8 + np.min( np.max(np.linalg.eigvals(H[1:,1:])), 0 )
            #     H[1:,1:] -= eps*np.eye(H.shape[0]-1)

            print('compute_Expectations finish')
            sys.stdout.flush()


            print('solve_inversion start')
            sys.stdout.flush()

            if constant_bias is None:
                delta_theta = mp.lu_solve(H, -grad)
                lambda2 = (grad.transpose() * delta_theta)[0]
                delta_theta = np.array(delta_theta)
            else:
                delta_theta = mp.lu_solve(H[1:, 1:], -grad[1:])
                lambda2 = (grad[1:].transpose() * delta_theta)[0]
                delta_theta = np.array(delta_theta)
                delta_theta = np.concatenate([np.array([mpf(0)]), delta_theta])

            print('solve_inversion finish')
            sys.stdout.flush()

            print('current_log_likelihood = ', current_log_likelihood)
            print('lambda_2 / 2 = ', lambda2/2)
            if lambda2 < 0:
                warnings.warn('lambda_2 < 0')
            if lambda2 / 2 < n*config.logpoly.theta_epsilon:
                print('%')
                break
                # [~, D] = eig(H);
                # diag(D)
                # # det(inv(H))
                # grad'*inv(H)*grad

            ## Line search
            lam = 1
            alpha = 0.49; beta = 0.5
            current_log_likelihood = _compute_log_likelihood(SS, np.concatenate(
                [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])]), n)

            while True:
                tmp_log_likelihood = _compute_log_likelihood(SS, np.concatenate(
                    [theta.reshape([-1, k + 1]), (theta_new + lam * delta_theta).reshape([1, -1])]), n)

                if tmp_log_likelihood < current_log_likelihood + alpha * lam * np.inner(grad, delta_theta):
                    print('+', end='')
                    sys.stdout.flush()
                    lam = lam * beta;
                else:
                    break

            if tmp_log_likelihood <= current_log_likelihood:
                # print('    number of iterations: ' + str(iteration+1))
                print('*')
                break
            
            theta_new = theta_new + lam * delta_theta
            current_log_likelihood = tmp_log_likelihood
            # print('theta_new = ', theta_new)

        logZ = mp_log_integral_exp(mp_compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]))
        return [theta_new, logZ, current_log_likelihood]
    
    def logpdf(self, x):
        p = mp_compute_poly(x, self.theta)
        return np.array( [ (a - self.logZ) for a in p ], dtype=float)
    
    def fit(self, x, n, plot=False):
        #if plot:
        #    if not os.path.isdir('./log'):
        #        os.mkdir('./log')

        for i in range(self.factors_count):
            # print('factor #' + str(i))
            # sys.stdout.flush()
            if i == 0:
                SS = mp_compute_SS(x, self.factor_degree)
                theta_new, self.logZ, self.current_log_likelihood = self._fit_new_factor(SS, n, constant_bias=1)
                #theta_new = np.ones([1,self.factor_degree+1])
                #self.logZ = 1
                #self.current_log_likelihood = 0
            else:
                SS = mp_compute_SS(x, self.factor_degree, self.theta)
                theta_new, self.logZ, self.current_log_likelihood = self._fit_new_factor(SS, n, constant_bias=None)
            self.theta = np.concatenate([self.theta.reshape([-1,self.factor_degree+1]), theta_new.reshape([1,-1])])
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


class LogpolyModelSelector:
    def __init__(self, list_factor_degrees):
        self.list_factor_degrees = list_factor_degrees

    def select_model(self, data):
        n_total = data.shape[0]


        if config.classifier.smart_validation:
            ind = np.argsort(data)
        else:
            ind = np.arange(n_total)
        index_train, index_validation = get_train_and_validation_index(ind)
        n_train = index_train.size

        avg_log_likelihoods = []
        logpoly_models = []
        for i, k in enumerate(self.list_factor_degrees):
            logpoly_models.append(Logpoly(factor_degree=k))
            logpoly_models[i].fit(data[index_train], n_train)
            avg_log_likelihoods.append(np.mean(logpoly_models[i].logpdf(data[index_validation])))

        best_index = np.argmax(avg_log_likelihoods)
        k = self.list_factor_degrees[best_index]
        best_logpoly_model = Logpoly(factor_degree=k)
        best_logpoly_model.fit(data, n_total)
        return best_logpoly_model


def _compute_log_likelihood(SS, theta, n):
    print('_compute_log_likelihood start')
    sys.stdout.flush()
    if len(theta.shape) == 1:
        theta = theta.reshape([1,-1])

    #logZ = log_integral_exp( compute_poly, theta, previous_critical_points)
    # ll = -n*logZ + np.inner(theta[-1,:].reshape([-1,]), SS.reshape([-1,]))
    print('    compute_logZ start')
    sys.stdout.flush()
    logZ = mp_log_integral_exp( mp_compute_poly, theta)
    print('    compute_logZ finish')
    sys.stdout.flush()
    ll = -n * logZ + np.inner(theta[-1, :].reshape([-1, ]), SS)
    print('_compute_log_likelihood finish')
    sys.stdout.flush()
    return ll