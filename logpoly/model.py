from .tools import compute_poly, log_integral_exp, compute_log_likelihood, compute_SS, scale_data #, TF_integrator
import config.logpoly
import config.general
import numpy as np
import scipy.integrate as integrate
# import os
# import matplotlib.pyplot as plt
import sys

# class LogpolyDataInterface:
#
#     def _extract_statistics(self, x, data_info):
#         self.n = len(x)
#         min_x = data_info.min_value
#         max_x = data_info.max_value
#         self.x = ((x - min_x) / (max_x - min_x)) * (
#                     0.9 * (config.logpoly.x_ubound - config.logpoly.x_lbound)) + config.logpoly.x_lbound + (
#                              0.05 * (config.logpoly.x_ubound - config.logpoly.x_lbound))
#
#     #def __init__(self):
#     #    x = np.load('./data/' + config.general.dataset_name + '.npy')
#     #    self._extract_statistics(x)
#
#     def __init__(self, x):
#         self._extract_statistics(x)
#
#     # def get_n(self):
#     #     return self.n

class Logpoly:

    def __init__(self, factor_degree, factors_count=1):
        self.factor_degree = factor_degree
        self.factors_count = factors_count
        self.theta = np.array([])


    def _fit_new_factor(self, SS, n, constant_bias = None):
        theta = self.theta

        k = self.factor_degree

        theta_new = np.zeros([k+1,])
        if constant_bias == None:
            theta_new[0] = 1
        else:
            theta_new[0] = constant_bias
        current_log_likelihood = None
        
        if theta is None:
            theta = np.array([], dtype=float)
        
        for iteration in range(config.logpoly.Newton_max_iter):
            
            # print('.',end='')
            # sys.stdout.flush()

            ## Compute sufficient statistics and constructing the gradient and the Hessian
            ESS = np.zeros([k+1,])
            logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]))
            
            for j in range(k+1):
                if len(theta) > 0:
                    def func(x):
                        return compute_poly(x, theta) * (x ** j) * np.exp(compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])])) - logZ)
                else:
                    def func(x):
                        return (x ** j) * np.exp(compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])])) - logZ)
                    
                ESS[j], _ = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)

            tmp = np.zeros([2*k+1, ])
            for j in range(2*k+1):
                if len(theta) > 0:
                    def func(x):
                        return (compute_poly(x, theta) ** 2) * (x ** j) * np.exp(compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])])) - logZ)
                else:
                    def func(x):
                        return (x ** j) * np.exp(compute_poly(x, np.concatenate(
                            [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])])) - logZ)
                
                tmp[j], _ = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)

            H = np.zeros([k+1,k+1])
            for i in range(k+1):
                for j in range(k+1):
                    H[i, j] = tmp[i+j]
                    
            H = -n*(H - np.matmul(ESS.reshape([-1,1]), ESS.reshape([1,-1])))
            grad = SS - n*ESS 
            
            if constant_bias is None:
                delta_theta = np.linalg.solve(H, -grad)
            else:
                delta_theta = np.linalg.solve(H[1:, 1:], -grad[1:])
                delta_theta = np.concatenate([np.array([0.0]), delta_theta])
            
            ## Line search
            lam = 1
            alpha = 0.49; beta = 0.5
            current_log_likelihood = compute_log_likelihood(SS, np.concatenate(
                [theta.reshape([-1, k + 1]), theta_new.reshape([1, -1])]), n)

            while compute_log_likelihood(SS, np.concatenate(
                    [theta.reshape([-1, k + 1]), (theta_new + lam * delta_theta).reshape([1, -1])]),
                                         n) < current_log_likelihood + alpha * lam * np.inner(grad, delta_theta):
                lam = lam * beta;

            if compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), n) <= current_log_likelihood:
                # print('    number of iterations: ' + str(iteration+1))
                break
            
            theta_new = theta_new + lam * delta_theta
        
        logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]))
        return [theta_new, logZ, current_log_likelihood]
    
    def logpdf(self, x):
        return compute_poly(x, self.theta) - self.logZ
    
    def fit(self, SS, n, plot=False):
        #if plot:
        #    if not os.path.isdir('./log'):
        #        os.mkdir('./log')

        for i in range(self.factors_count):
            # print('factor #' + str(i))
            sys.stdout.flush()
            if i == 0:
                theta_new, self.logZ, self.current_log_likelihood = self._fit_new_factor(SS, n, constant_bias=1)
                #theta_new = np.ones([1,self.factor_degree+1])
                #self.logZ = 1
                #self.current_log_likelihood = 0
            else:
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
        self.logpoly_models = []
        for k in list_factor_degrees:
            self.logpoly_models.append(Logpoly(factor_degree=k))

    def select_model(self, data):
        n_total = data.shape[0]
        ind = np.arange(n_total)
        np.random.shuffle(ind)
        n_train = n_total // 4

        avg_log_likelihoods = []
        for model in self.logpoly_models:
            print('  +')
            SS = compute_SS(data[:n_train], model.factor_degree)
            model.fit(SS, n_train)
            avg_log_likelihoods.append(np.mean(model.logpdf(data[n_train:])))

        return self.logpoly_models[np.argmax(avg_log_likelihoods)]
