import numpy as np


def emp_variogram(x,D,nh,h_bounds=0):
#   calculate the empirical variogram
    n = x.size
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij = np.subtract.outer(x, x)
    v = 0.5 * np.square(diff_ij)[triu]
    u=D[triu]

    if h_bounds==0:
        h = np.linspace(u.min(), u.max(), nh)
    else: 
        h = np.linspace(h_bounds[0], h_bounds[1], nh)

    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

    du = np.abs(u - h[:, None])
    w = np.exp(-np.square(2.68 * du / b) / 2)
    denom = np.nansum(w, axis=1)
    wv = w * v[None, :]
    num = np.nansum(wv, axis=1)
    ev = num / denom #empirical variogram
    return h,ev

def emp_variogramBinary(x,D,nh,h_bounds=0):
#   calculate the empirical variogram
    n = x.size
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij = np.subtract.outer(x, x)
    add_ij=np.add.outer(x,x)
    v = 0.5 * np.square(diff_ij)[triu]
    v_a=np.square(add_ij)[triu]
    index_keep=np.where(v_a>0)
    v=v[index_keep]
    u=D[triu]
    u=u[index_keep]
    if h_bounds==0:
        h = np.linspace(u.min(), u.max(), nh)
    else: 
        h = np.linspace(h_bounds[0], h_bounds[1], nh)

    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

    du = np.abs(u - h[:, None])
    w = np.exp(-np.square(2.68 * du / b) / 2)
    denom = np.nansum(w, axis=1)
    wv = w * v[None, :]
    num = np.nansum(wv, axis=1)
    ev = num / denom #empirical variogram
    return h,ev

def emp_variogram_vwise(x,D,nh,h_bounds=0):
    #calculate empirical variogram for each vertex

    n = x.size
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij = np.subtract.outer(x, x)
    u_full=D[triu]


    vw_ev=np.zeros([n,nh])
    vw_h=np.zeros([n,nh])
    for vert in range(n):
        
        diff_ij_vw=diff_ij[vert,:]
        diff_ij_vw=np.delete(diff_ij_vw,vert)
        v = 0.5 * np.square(diff_ij_vw)
        u=D[vert,:]
        u=np.delete(u,vert)
        if h_bounds==0:
            h = np.linspace(u.min(), u.max(), nh)
        else: 
            h = np.linspace(h_bounds[0], h_bounds[1], nh)
        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - h[:, None])
        w = np.exp(-np.square(2.68 * du / b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        ev = num / denom #empirical variogram
        vw_ev[vert,:]=ev
        vw_h[vert,:]=h
    
    return vw_h, vw_ev
    

def emp_searchlight(x,D,nh,min_dist,h_bounds=0):
    n = x.size
 

    vw_ev=np.zeros([n,nh])
    vw_h=np.zeros([n,nh])
    b=3
    
    
    for vert in range(n):
        idx = np.where(D[:,vert] <= min_dist)
        
        D_sl=D[np.ix_(idx[0],idx[0])]
        x_sl=x[idx[0]]
        diff_ij = np.subtract.outer(x_sl, x_sl)
        triu = np.triu_indices(len(idx[0]), k=1)  # upper triangular inds
        v = 0.5 * np.square(diff_ij)[triu]
        u=D_sl[triu]


        if h_bounds==0:
            h = np.linspace(u.min(), u.max(), nh)
        else: 
            h = np.linspace(h_bounds[0], h_bounds[1], nh)


    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

        du = np.abs(u - h[:, None])
        w = np.exp(-np.square(2.68 * du / b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        ev = num / denom #empirical variogram
        vw_ev[vert,:]=ev
        vw_h[vert,:]=h

    return vw_h, vw_ev


def emp_variogram_vwise_SeedFC(x,D,nh,h_bounds=0):
    #calculate empirical variogram for each vertex

    n = x.shape[0]
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3




    vw_ev=np.zeros([n,nh])
    vw_h=np.zeros([n,nh])
    for vert in range(n):
        diff_ij = np.subtract.outer(x[:,vert], x[:,vert])
        u_full=D[triu]
        diff_ij_vw=diff_ij[vert,:]
        diff_ij_vw=np.delete(diff_ij_vw,vert)
        v = 0.5 * np.square(diff_ij_vw)
        u=D[vert,:]
        u=np.delete(u,vert)

        if h_bounds:
        	h = np.linspace(u.min(), u.max(), nh)
        else: 
            h = np.linspace(h_bounds[0], h_bounds[1], nh)


        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - h[:, None])
        w = np.exp(-np.square(2.68 * du / b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        ev = num / denom #empirical variogram
        vw_ev[vert,:]=ev
        vw_h[vert,:]=h
    
    return vw_h, vw_ev

def emp_cross_variogram(x1,x2,D,nh,h_bounds=0):
	#   calculate the empirical cross-variogram
    n = x1.size
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij_1 = np.subtract.outer(x1, x1)
    diff_ij_2 = np.subtract.outer(x2, x2)
    #v1 = 0.5 * np.square(diff_ij_1)[triu]
    #v2 = 0.5 * np.square(diff_ij_2)[triu]
    v1 = diff_ij_1[triu]
    v2 = diff_ij_2[triu]
    v=0.5* v1*v2
    u=D[triu]

    if h_bounds==0:
        h = np.linspace(u.min(), u.max(), nh)
    else: 

        h = np.linspace(h_bounds[0], h_bounds[1], nh)

    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

    du = np.abs(u - h[:, None])
    w = np.exp(-np.square(2.68 * du / b) / 2)
    denom = np.nansum(w, axis=1)
    wv = w * v[None, :]
    num = np.nansum(wv, axis=1)
    ev = num / denom #empirical variogram
    return h,ev
    
    

def emp_variogram_conn(x,D,nh,h_bounds=0):
#   calculate the empirical variogram
    n = x.shape[0]
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij = 1-x
    v = 0.5 * np.square(diff_ij)[triu]
    u=D[triu]

    if h_bounds==0:
        h = np.linspace(u.min(), u.max(), nh)
    else: 
        h = np.linspace(h_bounds[0], h_bounds[1], nh)

    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

    du = np.abs(u - h[:, None])
    w = np.exp(-np.square(2.68 * du / b) / 2)
    denom = np.nansum(w, axis=1)
    wv = w * v[None, :]
    num = np.nansum(wv, axis=1)
    ev = num / denom #empirical variogram
    return h,ev



def emp_variogram_vwise_conn(x,D,nh,h_bounds=0):
    #calculate empirical variogram for each vertex

    n = x.shape[0]
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij = 1-x
    u_full=D[triu]


    vw_ev=np.zeros([n,nh])
    vw_h=np.zeros([n,nh])
    for vert in range(n):
        
        diff_ij_vw=diff_ij[vert,:]
        diff_ij_vw=np.delete(diff_ij_vw,vert)
        v = 0.5 * np.square(diff_ij_vw)
        u=D[vert,:]
        u=np.delete(u,vert)
        if h_bounds==0:
            h = np.linspace(u.min(), u.max(), nh)
        else: 
            h = np.linspace(h_bounds[0], h_bounds[1], nh)
        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - h[:, None])
        w = np.exp(-np.square(2.68 * du / b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        ev = num / denom #empirical variogram
        vw_ev[vert,:]=ev
        vw_h[vert,:]=h
    
    return vw_h, vw_ev

def emp_cross_variogram_conn(x1,x2,D,nh,h_bounds=0):
	#   calculate the empirical cross-variogram
    n = x1.shape[0]
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij_1 = np.subtract.outer(x1, x1)
    diff_ij_2 = np.subtract.outer(x2, x2)
    #v1 = 0.5 * np.square(diff_ij_1)[triu]
    #v2 = 0.5 * np.square(diff_ij_2)[triu]
    v1 = diff_ij_1[triu]
    v2 = diff_ij_2[triu]
    v=0.5* v1*v2
    u=D[triu]

    if h_bounds==0:
        h = np.linspace(u.min(), u.max(), nh)
    else: 

        h = np.linspace(h_bounds[0], h_bounds[1], nh)

    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

    du = np.abs(u - h[:, None])
    w = np.exp(-np.square(2.68 * du / b) / 2)
    denom = np.nansum(w, axis=1)
    wv = w * v[None, :]
    num = np.nansum(wv, axis=1)
    ev = num / denom #empirical variogram
    return h,ev
    
def emp_crossvariogram_vwise(x1,x2,D,nh,h_bounds=0):
    #calculate empirical variogram for each vertex

    u_full=D[triu]

    n = x1.shape[0]
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij_1 = np.subtract.outer(x2, x2)
    diff_ij_2 = np.subtract.outer(x2, x2)

    vw_ev=np.zeros([n,nh])
    vw_h=np.zeros([n,nh])
    for vert in range(n):
        
        diff_ij_vw_1=diff_ij_1[vert,:]
        diff_ij_vw_1=np.delete(diff_i_1j_vw,vert)
        diff_ij_vw_2=diff_ij_2[vert,:]
        diff_ij_vw_2=np.delete(diff_ij_2_vw,vert)
        #v = 0.5 * np.square(diff_ij_vw)
        #v=0.5* v1*v2
        v=0.5* diff_ij_vw_1*diff_ij_vw_2
        u=D[vert,:]
        u=np.delete(u,vert)
        if h_bounds==0:
            h = np.linspace(u.min(), u.max(), nh)
        else: 
            h = np.linspace(h_bounds[0], h_bounds[1], nh)
        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - h[:, None])
        w = np.exp(-np.square(2.68 * du / b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        ev = num / denom #empirical variogram
        vw_ev[vert,:]=ev
        vw_h[vert,:]=h
    
    return vw_h, vw_ev


def emp_crossvariogram_vwise_conn(x1,x2,D,nh,h_bounds=0):
    #calculate empirical variogram for each vertex

    

    n = x1.shape[0]
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij_1 = 1-x1
    diff_ij_2 = np.subtract.outer(x2, x2)

    vw_ev=np.zeros([n,nh])
    vw_h=np.zeros([n,nh])
    for vert in range(n):
        diff_ij_vw_1=diff_ij_1[vert,:]
        diff_ij_vw_1=np.delete(diff_ij_vw_1,vert)
        diff_ij_vw_2=diff_ij_2[vert,:]
        diff_ij_vw_2=np.delete(diff_ij_vw_2,vert)
        #v = 0.5 * np.square(diff_ij_vw)
        #v=0.5* v1*v2
        v=0.5* diff_ij_vw_1*diff_ij_vw_2
        u=D[vert,:]
        u=np.delete(u,vert)
        if h_bounds==0:
            h = np.linspace(u.min(), u.max(), nh)
        else: 
            h = np.linspace(h_bounds[0], h_bounds[1], nh)
        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - h[:, None])
        w = np.exp(-np.square(2.68 * du / b) / 2)
        denom = np.nansum(w, axis=1)
        wv = w * v[None, :]
        num = np.nansum(wv, axis=1)
        ev = num / denom #empirical variogram
        vw_ev[vert,:]=ev
        vw_h[vert,:]=h
    
    return vw_h, vw_ev



#from libpysal.lib import weights
#from pysal.explore.esda import Moran
#from libpysal import weights
#from esda.moran import Moran

# Create inverse distance weight matrix assuming D is an NxN pairwise inter-parcel geodesic distance matrix
#D[np.eye(D.shape[0]).astype(bool)] = 1 # temporary, for inversion
#D = 1. / D
#D[np.eye(D.shape[0]).astype(bool)] = 0 # mask diagonal elements
#W = weights.full2W(D) # create weights object

# Compute moran's I for the length-N brain map x
#mi = Moran(x, W).I

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))