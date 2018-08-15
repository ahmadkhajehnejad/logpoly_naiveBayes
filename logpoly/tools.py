import config.logpoly
import numpy as np
import tensorflow as tf
from scipy import integrate

def log_sum_exp(a, axis=0, keepdims=False):
    mx = np.max( a, axis = axis, keepdims=keepdims)
    tile_shape = np.ones([len(a.shape),], dtype=int)
    tile_shape[axis] = a.shape[axis]
    tmp_shape = [i for i in a.shape]
    tmp_shape[axis] = 1
    res = mx + np.log(np.sum( np.exp(a-np.tile(mx.reshape(tmp_shape),tile_shape)), axis=axis, keepdims=keepdims))
    return res
 
def compute_poly(x, theta):
    if len(theta.shape) == 1:
        theta = theta.reshape([1,-1])
    if type(x) != np.ndarray:
        if type(x) == list:
            x = np.array(x)
        else:
            x = np.array([x])
        
    k = theta.shape[1] - 1
    t = theta.shape[0]
    n = x.size
    
    theta3d = np.tile(np.expand_dims(theta,2), [1,1,n])
    x3d = np.tile(x.reshape([1,1,n]), [t,k+1,1])
    exp3d = np.tile(np.arange(k+1).reshape([1,k+1,1]),[t,1,n])
    
    return np.prod(np.sum( (x3d ** exp3d) * theta3d, axis=1), axis=0)


def compute_SS(x, theta=None):

    n = x.size
    k = config.logpoly.factor_degree
    
    if theta is None:
        p = np.ones([n,1])
    elif theta.shape[0] == 0:
        p = np.ones([n,1])
    else:
        p = compute_poly(x, theta)
                
    exponent = np.tile(np.arange(k+1).reshape([1,-1]), [n,1])
    base = np.tile(x.reshape([-1,1]), [1,k+1])
    coef = np.tile(p.reshape([-1,1]), [1,k+1])
    
    return np.sum( (base ** exponent) * coef , axis=0)

'''
def log_integral_exp( log_func, theta, critical_points, return_new_critical_points=False):
    x_lbound, x_ubound = config.logpoly.x_lbound, config.logpoly.x_ubound
    if len(theta.shape) > 1:
        theta = theta[-1,:]
    theta = theta.reshape([-1,])
    d = theta.size - 1
    derivative_poly_coeffs = np.flip(theta[1:] * np.arange(1,d+1), axis=0)
        
    r = np.roots(derivative_poly_coeffs)
    r = r.real[r.imag < 1e-10]
    r = np.unique(r[ (r >= x_lbound) & (r <= x_ubound)])
    
    if critical_points is None:
        new_critical_points = r
    elif critical_points.shape[0] == 0:
        new_critical_points = r
    else:
        new_critical_points = critical_points.copy()
        if r.size > 0:
            new_critical_points = np.unique(np.concatenate([new_critical_points, r]))
    
    if new_critical_points.size > 0:    
        br_points = np.unique(np.concatenate([np.array([x_lbound]), new_critical_points, np.array([x_ubound])]))
    else:
        br_points = np.array([x_lbound, x_ubound])    
    
    buff = np.zeros( [br_points.size -1,]);
    for i in range(br_points.size - 1):
        p1 = br_points[i]
        p2 = br_points[i+1]
        l = p2 - p1
        parts = 100000
        points = np.arange(p1,p2+1e-10,l/parts)
        
        f = log_func(points, theta)        
        f = np.concatenate( [ f, f[1:-1] ] )

        buff[i] = log_sum_exp(f) + np.log(l/parts) - np.log(2)
        

    if return_new_critical_points:
        return [log_sum_exp(buff), new_critical_points]
    else:
        return log_sum_exp(buff);
'''

def log_integral_exp( log_func, theta):
    x_lbound, x_ubound = config.logpoly.x_lbound, config.logpoly.x_ubound
    
    
    if len(theta.shape) == 1:
        theta = theta.reshape([1,-1])
    all_theta = theta[0,:].reshape([-1]).copy()
    d = all_theta.size - 1
    
    
    for i in range(1,len(theta)):
        tmp = np.matmul(theta[i,:].reshape([-1,1]), all_theta.reshape([1,-1]))
        all_theta = np.zeros(all_theta.size + d)
        for i1 in range(tmp.shape[0]):
            for i2 in range(tmp.shape[1]):
                all_theta[i1+i2] += tmp[i1,i2]
    
    d_all = all_theta.size - 1
    derivative_poly_coeffs = np.flip(all_theta[1:] * np.arange(1,d_all+1), axis=0)
        
    r = np.roots(derivative_poly_coeffs)
    r = r.real[r.imag < 1e-10]
    r = np.unique(r[ (r >= x_lbound) & (r <= x_ubound)])
    
    if r.size > 0:
        br_points = np.unique(np.concatenate([np.array([x_lbound]), r.reshape([-1]), np.array([x_ubound])]))
    else:
        br_points = np.array([x_lbound, x_ubound])    
    
    buff = np.zeros( [br_points.size -1,]);
    for i in range(br_points.size - 1):
        p1 = br_points[i]
        p2 = br_points[i+1]
        l = p2 - p1
        parts = 1000
        points = np.arange(p1,p2+1e-10,l/parts)
        
        f = log_func(points, theta)        
        f = np.concatenate( [ f, f[1:-1] ] )

        buff[i] = log_sum_exp(f) + np.log(l/parts) - np.log(2)
    return log_sum_exp(buff);

def compute_log_likelihood(SS, theta, n):
    if len(theta.shape) == 1:
        theta = theta.reshape([1,-1])
    #if len(theta.shape) > 1:
    #    theta = theta[-1,:]
    
    #logZ = log_integral_exp( compute_poly, theta, previous_critical_points)
    logZ = log_integral_exp( compute_poly, theta)
    ll = -n*logZ + np.inner(theta[-1,:].reshape([-1,]), SS.reshape([-1,]))
    
    return ll


class TF_integrator:
    
    sess = tf.Session()
    
    def integrate(func, lbound, ubound):
        def integrated( f, x ):
            return tf.map_fn( lambda y: tf.py_func( 
                       lambda z: integrate.quad( f, lbound, z )[ 0 ], [ y ], tf.float64 ), x )
        
        x = tf.constant( [ ubound ], dtype = tf.float64 )
        
        _int = integrated(func, x)
    
        res = TF_integrator.sess.run( _int )
        return res