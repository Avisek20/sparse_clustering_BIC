import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def bic_fcm(data, centers, u, w):
    N, dim = data.shape
    k = centers.shape[0]
    var = cdist(data, centers, metric='sqeuclidean').sum() * (1 / (k * dim * (N - 1)))
    return ((np.log(u).sum()) - (0.5 * dim * k * (N - 1))
        - (0.5 * k * dim * N * np.log(2 * np.pi * var)) - (0.5 * k * (N + dim) * np.log(N)))


def select_s(data, m=2, num_s=30, n_clusters=1, max_iter=300, n_init=30, tol=1e-15, n_jobs=1):
    svals = np.exp(np.linspace(np.log(1.2), np.log((data.shape[1]**0.5) * 0.9), num_s))

    u = np.zeros((num_s, n_clusters, data.shape[0]))
    w = np.zeros((num_s, data.shape[1]))
    centers = np.zeros((num_s, n_clusters, data.shape[1]))
    cost = np.zeros((num_s), dtype=np.float64)
    list_bic = np.zeros((num_s))
    # For each value of s, cluster the data set
    for s in range(num_s):
        #print(s, '/', num_s)
        # cluster the data set
        u[s], w[s], centers[s], cost[s] = wfcm(
            data, m=m, n_clusters=n_clusters, s=svals[s], max_iter=max_iter, n_init=n_init, tol=tol
        )
        list_bic[s] = bic_fcm(data, centers[s], u[s], w[s])
    selected_idx = list_bic.argmax()

    return u, w, centers, cost, svals, list_bic, selected_idx


def update_weights(data, um, tot_var, s, thresh=1e-8):
    centers = um.dot(data) / um.sum(axis=1)[:,None]
    intra_var = np.zeros((data.shape[1]))
    for j in range(um.shape[0]):
        intra_var += (um[j,:][:,None] * ((data -  centers[j]) ** 2)).sum(axis=0)

    a = tot_var - intra_var
    w = np.sign(a) * a / np.fmax(np.linalg.norm(a), np.finfo(float).eps)
    # Pick delta so that the l1 norm of w is s
    # The min possible value for delta is 0,
    # and the max value is set to the 2nd maximum of a
    # since that sets w to a vector with one 1 and the rest 0s.
    low = 0
    high = a.max()#np.sort(a)[-2]
    while (high-low) > thresh:
        delta = 0.5 * (high + low)
        w = np.sign(a) * np.fmax(a - delta, 0)
        w = w / np.fmax(
            np.linalg.norm(w), np.finfo(float).eps
        )
        if np.abs(np.sum(w) - s) < thresh:
            break
        elif np.sum(w) < s:
            high = delta
        else:
            low = delta
    return w


def wfcm(data, n_clusters=1, s=None, m=2, n_init=30, max_iter=300, tol=1e-15, ret_cost=False):
    tot_var = data.shape[0] * data.var(axis=0)

    max_cost = 0
    for iter_init in range(n_init):
        # Initialize u using kmeans
        centers = KMeans(n_clusters=n_clusters, n_init=n_init).fit(data).cluster_centers_
        dist = cdist(centers, data, metric='sqeuclidean')
        u = ( 1 / dist ) ** ( 1 / (m-1) )
        um = ( u / u.sum(axis=0) ) ** m

        # initialise weights
        w = np.ones((data.shape[1])) / data.shape[1]**0.5

        for var_iter in range(max_iter):

            # Update center
            Z = data[:,w>0] * (w[w>0] ** 0.5)
            centers = um.dot(Z) / um.sum(axis=1)[:,None]

            # Update mem
            dist = cdist(centers, Z, metric='sqeuclidean')
            u = ( 1 / dist ) ** ( 1 / (m-1) )
            um = ( u / u.sum(axis=0) ) ** m

            # Update w
            prev_w = np.array(w)
            w = update_weights(data, um, tot_var, s, thresh=tol)

            if np.linalg.norm(w - prev_w) < tol:
                break

        cost = 0
        centers = um.dot(data) / um.sum(axis=1)[:,None]
        intra_var = np.zeros((data.shape[1]))
        for j in range(n_clusters):
            intra_var += (um[j,:][:,None] * ((data -  centers[j]) ** 2)).sum(axis=0)
        cost = ((w) * (tot_var - intra_var)).sum()

        if cost > max_cost:
            max_cost = cost
            maxcost_u = np.array(u / u.sum(axis=0))
            maxcost_w = np.array(w)
            maxcost_centers = np.array(centers)

    if ret_cost:
        return max_cost
    else:
        return maxcost_u, maxcost_w, maxcost_centers, max_cost


if __name__ == '__main__':

    datasets = [
        '2d-4c-2x2.npz', '2d-6c-3x2.npz', '2d-8c-4x2.npz', '2d-15c-5x3.npz'
    ]

    max_iter = 20
    n_init = 20
    tol = 1e-6
    n_s = 10
    m = 1.2
    n_runs = 5

    for adataset in range(len(datasets)):
        print('Dataset: ', datasets[adataset])

        tmp = np.load('datasets/'+datasets[adataset])
        X = tmp['X']
        y = tmp['y']
        k = len(np.unique(y))
        print('Data size =', X.shape)
        print('# clusters =', k)
        
        for i_runs in range(n_runs):
            u, w, centers, cost, svals, BIC, selected_idx = select_s(
                X, m=m, num_s=n_s, n_clusters=k, max_iter=max_iter, n_init=n_init, tol=tol, n_jobs=10
            )
    
            #print('svals', svals)
            #print('selected s:', svals[selected_idx])
            #print('w:', w[selected_idx])
            #print('cost:', cost[selected_idx])
            #print('BIC:', BIC)
            #print(BIC.argmax())
            from sklearn.metrics import adjusted_rand_score as ARI
            ari = ARI(y, u[selected_idx].argmax(axis=0))
            print('ARI =', ari)

        #break
        
