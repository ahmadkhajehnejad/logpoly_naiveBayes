from .tools import compute_poly, log_integral_exp, compute_log_likelihood, compute_SS
import config.logpoly
import config.general
import numpy as np
import scipy.integrate as integrate

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
    
    def fit_new_factor(self):
        theta = self.theta
        critical_points = self.critical_points
        
        SS = self.interface.get_SS(theta)
        n = self.interface.get_n()
        k = config.logpoly.factor_degree
        
        theta_new = np.zeros([k+1,])
        current_log_likelihood = None
        
        if theta == None:
            theta = np.array([], dtype=float)
        
        for iteration in range(config.logpoly.Newtor_max_iter):
            
            ## Compute sufficient statistics and constructing the gradient and the Hessian
            ESS = np.zeros([k,]);
            logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), critical_points)
            
            for j in range(k+1):
                def func(x):
                    return compute_poly(x, theta) * (x ** j) * np.exp(compute_poly(x,theta_new) - logZ)
                
                ESS[j] = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)
    
            tmp = np.concatenate([ ESS, np.zeros(k)])
            for j in range(k+1,2*k+1):
                def func(x):
                    return (compute_poly(x, theta)**2) * (x ** j) * np.exp(compute_poly(x,theta_new) - logZ)
                
                tmp[j] = integrate.quad(func, config.logpoly.x_lbound, config.logpoly.x_ubound)
                
            H = np.zeros([k+1,k+1])
            for i in range(k+1):
                for j in range(k+1):
                    H[i,j] = tmp[i+j]
    
            H = -n*(H - np.matmul(ESS.reshape([-1,1]), ESS.reshape([1,-1])))
            grad = SS - n*ESS
            delta_theta = np.linalg.solve(H,-grad)
            
            ## Line search
            lam = 1
            alpha = 0.49; beta = 0.5
            current_log_likelihood = compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), critical_points, n)
            
            while compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), critical_points, n) < current_log_likelihood + alpha*lam*np.inner(grad,delta_theta):
                lam = lam * beta;
                
            if compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k+1]), (theta_new + lam*delta_theta).reshape([1,-1])]), critical_points, n) <= current_log_likelihood:
                break
            
            theta_new = theta_new + lam * delta_theta
    
        
        [logZ, new_critical_points] = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k+1]), theta_new.reshape([1,-1])]), critical_points, return_new_critical_points=True)
        return [theta_new, logZ, new_critical_points]
    
    
    
    def fit(self):
        self.theta = np.array([])
        self.critical_points = np.array([])
        for i in range(config.logpoly.num_factors):
            theta_new, self.logZ, self.critical_points = self.fit_new_factor()
            self.theta = np.concatenate([self.theta.reshape([-1,config.logpoly.factor_degree+1]), theta_new.reshape([1,-1])])