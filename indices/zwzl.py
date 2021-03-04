# Paper Source: Y. Zhang, W. Wang, X. Zhang, and Y. Li.
# A cluster validity index for fuzzy clustering.
# Information Sciences, 178(4):1205 – 1218, 2008.

# Min number of clusters = 2
# The time complexity is O(nk), where n is the number of data points
# and k is the number of clusters.
# To estimate the number of clusters: Use min of ZWZL


import numpy as np
from scipy.spatial.distance import cdist


def ZWZL(data, centers, m=2):
    dist = cdist(centers, data, metric='sqeuclidean')
    u = ( 1 / dist ) ** ( 1 / (m-1) )
    u = u / u.sum(axis=0)

    beta = data.shape[0] / ((data - data.mean(axis=0)) ** 2).sum()
    k = centers.shape[0]
    var = ((u * (1 - np.exp(-beta * dist)) ** 0.5).sum(axis=1) / u.sum(axis=1)).sum() * (((k+1) / (k-1)) ** 0.5)
    sep = np.zeros((k*(k-1)))
    sepc = 0
    for j1 in range(k):
        for j2 in range(j1+1,k):
            sep[sepc] = max(np.fmin(u[j1], u[j2]))
            sepc += 1
    sep = 1 - sep.max()

    return var, sep


if __name__ == '__main__':
    import time
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    var = np.zeros((maxK-minK))
    sep = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=5, max_iter=100, tol=1e-7
        ).fit(data)
        var[k-minK], sep[k-minK] = ZWZL(
            data, km1.cluster_centers_
        )
    var = var / var.max()
    sep = sep / sep.max()

    index = var / sep
    print('index:', index)
    print('est k:', index.argmin()+minK)
