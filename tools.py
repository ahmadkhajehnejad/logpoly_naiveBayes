import numpy as np

def log_sum_exp(a, d):
     mx = np.max(a,axis = d);
     rp = ones(1,len(a.shape);
     rp(d) = size(a,d);
     s = mx + log(sum( exp(a-repmat(mx,rp)), d));
     s = double(s);
