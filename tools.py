import numpy as np

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
    d = theta.shape[1] - 1
    k = theta.shape[0]
    n = x.size
    
    theta3d = np.tile(np.expand_dims(theta,2), [1,1,n])
    x3d = np.tile(x.reshape([1,1,n]), [k,d+1,1])
    exp3d = np.tile(np.arange(d+1).reshape([1,d+1,1]),[k,1,n])
    
    return np.prod(np.sum( (x3d ** exp3d) * theta3d, axis=1), axis=0)


def compute_SS(x, theta, d=None):

    n = x.size
    
    if theta == None:
        p = np.ones([n,])
    elif theta.shape[0] == 0:
        p = np.ones([n,1])
    else:
        d = theta.shape[1] - 1
        p = compute_poly(x, theta)
                
    exponent = np.tile(np.arange(d+1).reshape([1,-1]), [n,1])
    base = np.tile(x.reshape([-1,1]), [1,d+1])
    coef = np.tile(p.reshape([-1,1]), [1,d+1])
    
    return np.sum( (base ** exponent) * coef , axis=0)


def log_integral_exp( log_func, theta, critical_points, x_lbound, x_ubound ):
    if len(theta.shape) > 1:
        theta = theta[-1,:]
    theta = theta.reshape([-1,])
    d = theta.size - 1
    derivative_poly_coeffs = np.flip(theta[1:] * np.arange(1,d), axis=0)
    
    r = np.roots(derivative_poly_coeffs)
    r = r.real[r.imag < 1e-10]
    r = r[r >= x_lbound & r <= x_ubound]
    
    if critical_points == None:
        critical_points = r
    elif critical_points.shape[0] == 0:
        critical_points = r
    else:
        critical_points = np.unique(np.concatenate[critical_points, r])
    
    br_points = np.unique(np.concatenate([x_lbound, critical_points, x_ubound]))
    
    buff = np.zeros( [br_points.size -1,]);
    for i in range(br_points.size - 1):
        p1 = br_points[i]
        p2 = br_points[i+1]
        l = p2 - p1
        parts = 200
        points = np.arange(p1,p2,l/parts)
        
        f = log_func(points, theta)
        
        f = np.concatenate( [ f, f[1:-1] ] )
        buff[i] = log_sum_exp(f) + np.log(l/parts) - np.log(2)
    
    return log_sum_exp(buff);
    
def compute_log_likelihood(SS, theta, previous_critical_points, n, x_lbound, x_ubound, return_logZ=False):
    if len(theta.shape) > 1:
        theta = theta[-1,:]
    
    logZ = log_integral_exp( compute_poly, theta, previous_critical_points, x_lbound, x_ubound)
    
    ll = -n*logZ + np.inner(theta.reshape([-1,]), SS.reshape([-1,]))
    
    if return_logZ:
        return [ll, logZ]
    else:
        return ll