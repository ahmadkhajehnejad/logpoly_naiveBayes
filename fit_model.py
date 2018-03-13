from tools import compute_poly, log_integral_exp, compute_log_likelihood
import config
import numpy as np
import scipy.integrate as integrate

def add_new_factor(interface, theta=None, critical_points = None):
    SS = interface.get_SS(theta)
    n = interface.get_n()
    k = config.factor_degree
    
    theta_new = np.zeros([k+1,])
    current_log_likelihood = None
    
    if theta == None:
        theta = np.array([], dtype=float)
    
    for iteration in range(config.Newtor_max_iter):
        
        ## Compute sufficient statistics and constructing the gradient and the Hessian
        ESS = np.zeros([k,]);
        logZ = log_integral_exp(compute_poly, np.concatenate([theta.reshape([-1,k]), theta_new.reshape([1,-1])]), critical_points)
        
        for j in range(k+1):
            def func(x):
                return compute_poly(x, theta) * (x.reshape([-1,]) ** j) * np.exp(compute_poly(x,theta_new) - logZ)
            
            ESS[j] = integrate.quad(func, config.x_lbound, config.x_ubound)

        tmp = np.concatenate([ ESS, np.zeros(k)])
        for j in range(k+1,2*k+1):
            def func(x):
                return (compute_poly(x, theta)**2) * (x.reshape([-1,]) ** j) * np.exp(compute_poly(x,theta_new) - logZ)
            
            tmp[j] = integrate.quad(func, config.x_lbound, config.x_ubound)
            
        H = np.zeros([k,k])
        for i in range(k):
            for j in range(k):
                H[i,j] = tmp[i+j]

        H = -n*(H - np.matmul(ESS.reshape([-1,1]), ESS.reshape([1,-1])))
        grad = SS - n*ESS
        delta_theta = np.linalg.solve(H,-grad)
        
        ## Line search
        lam = 1
        alpha = 0.49; beta = 0.5
        current_log_likelihood = compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k]), theta_new.reshape([1,-1])]), critical_points, n)
        
        while compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k]), (theta_new + lam*delta_theta).reshape([1,-1])]), critical_points, n) < current_log_likelihood + alpha*lam*np.inner(grad,delta_theta):
            lam = lam * beta;
            
        if compute_log_likelihood(SS, np.concatenate([theta.reshape([-1,k]), (theta_new + lam*delta_theta).reshape([1,-1])]), critical_points, n) <= current_log_likelihood:
            break
        
        theta_new = theta_new + lam * delta_theta

    theta = np.concatenate([theta.reshape([-1,k]), theta_new.reshape([1,-1])])
    [logZ, critical_points] = log_integral_exp(compute_poly, theta, critical_points, return_new_critical_points=True)
