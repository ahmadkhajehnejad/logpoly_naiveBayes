from .tools import compute_poly, log_integral_exp, compute_log_likelihood, compute_SS, tf_integrate
import config.logpoly
import config.general
import numpy as np
import scipy.integrate as integrate
import os
import matplotlib.pyplot as plt

class Interface:
    def __init__(self):
        x = np.load('./data/' + config.general.dataset_name + '.npy')
        self.n = len(x)
        min_x = np.min(x,axis=0)
        max_x = np.max(x,axis=0)
        self.x = ( (x - min_x) / (max_x - min_x) ) * (0.9 * (config.logpoly.x_ubound - config.logpoly.x_lbound)) + config.logpoly.x_lbound + (0.05 * (config.logpoly.x_ubound - config.logpoly.x_lbound))
        
    def get_SS(self, theta):
        return compute_SS(self.x,theta)
    
    def get_n(self):
        return self.n


class Logpoly:
    
    def __init__(self):
        self.interface = Interface()
    
    def fit_new_factor(self, constant_bias = None):
        theta = self.theta
        
        SS = self.interface.get_SS(theta)
        n = self.interface.get_n()
        k = config.logpoly.factor_degree
        
        theta_new = np.zeros([k+1,])
        if constant_bias == None:
            theta_new[0] = 1
        else:
            theta_new[0] = constant_bias
        current_log_likelihood = None
        
        if theta is None:
            theta = np.array([], dtype=float)
        
        for iteration in range(config.logpoly.Newton_max_iter):
            
            print('.',end='')
            ## Compute sufficient statistics and constructing the gradient and the Hessian
            ESS = np.zeros([k+1,]);
            #logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), critical_points)
            logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]))
                        
            '''
            def func(x):
                return np.exp(compute_poly(x,np.concatenate([theta.reshape([-1,k+1]),theta_new.reshape([1,-1])])) - logZ)
            tmp,_ = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)
            print(tmp)
            '''
            
            for j in range(k+1):
                if len(theta) > 0:
                    def func(x):
                        return compute_poly(x, theta) * (x ** j) * np.exp(compute_poly(x,np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])])) - logZ)
                else:
                    def func(x):
                        return (x ** j) * np.exp(compute_poly(x,np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])])) - logZ)
                if config.logpoly.use_tf:
                    ESS[j] = tf_integrate(func, config.logpoly.x_lbound, config.logpoly.x_ubound)
                else:
                    ESS[j],_ = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)
                
    
            tmp = np.zeros([2*k+1,])
            for j in range(2*k+1):
                if len(theta) > 0:
                    def func(x):
                        return (compute_poly(x, theta)**2) * (x ** j) * np.exp(compute_poly(x,np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])])) - logZ)
                else:
                    def func(x):
                        return (x ** j) * np.exp(compute_poly(x,np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])])) - logZ)
                
                tmp[j],_ = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)
                
            H = np.zeros([k+1,k+1])
            for i in range(k+1):
                for j in range(k+1):
                    H[i,j] = tmp[i+j]

            #print('theta : {}    H : {}    logZ : {}'.format(np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), H, logZ))                
                    
            H = -n*(H - np.matmul(ESS.reshape([-1,1]), ESS.reshape([1,-1])))
            grad = SS - n*ESS 
            
            if constant_bias == None:
                delta_theta = np.linalg.solve(H,-grad)
            else:
                delta_theta = np.linalg.solve(H[1:,1:], -grad[1:])
                delta_theta = np.concatenate([np.array([0.0]), delta_theta])
            
            ## Line search
            lam = 1
            alpha = 0.49; beta = 0.5
            #current_log_likelihood = compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), critical_points, n)
            current_log_likelihood = compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), n)
            
            #print('current_log_likelihood = {}'.format(current_log_likelihood))
            
            #while compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), critical_points, n) < current_log_likelihood + alpha*lam*np.inner(grad,delta_theta):
            while compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), n) < current_log_likelihood + alpha*lam*np.inner(grad,delta_theta):
                lam = lam * beta;
        
            #print('lam = {}'.format(lam))
            #print(lam*delta_theta)
            #if compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), critical_points, n) <= current_log_likelihood:
            if compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), n) <= current_log_likelihood:
                print('    number of iterations: ' + str(iteration+1))
                break
            
            theta_new = theta_new + lam * delta_theta
        
        logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]))
        return [theta_new, logZ, current_log_likelihood]
    
    def logpdf(self, x):
        return compute_poly(x, self.theta) - self.logZ
    
    def fit(self, plot=False):
        if plot:
            if not os.path.isdir('./log'):
                os.mkdir('./log')
                
        self.theta = np.array([])
        
        for i in range(config.logpoly.num_factors):
            print('factor #' + str(i))
            if i == 0:
                theta_new, self.logZ, self.current_log_likelihood = self.fit_new_factor(constant_bias=1)
                #theta_new = np.ones([1,config.logpoly.factor_degree+1])
                #self.logZ = 1
                #self.current_log_likelihood = 0
            else:
                theta_new, self.logZ, self.current_log_likelihood = self.fit_new_factor(constant_bias=None)
            self.theta = np.concatenate([self.theta.reshape([-1,config.logpoly.factor_degree+1]), theta_new.reshape([1,-1])])
            print(self.theta)
            print(self.current_log_likelihood)
            if plot:
                plt.cla()
                
                l = config.logpoly.x_ubound - config.logpoly.x_lbound
                plt.hist(self.interface.x,np.arange(config.logpoly.x_lbound + (l/200), config.logpoly.x_ubound, l/100), density=True)
                
                x = np.arange(config.logpoly.x_lbound, config.logpoly.x_ubound, 0.001)
                p = np.exp(self.logpdf(x))
                plt.plot(x,p)
                plt.ylim([0,5])
                plt.text(0,4,'log likelihood:\n      ' + str(self.current_log_likelihood))
                plt.show()
                plt.savefig('./log/' + str(i) + '.png')
                plt.close()