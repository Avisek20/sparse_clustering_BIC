# Improved Efficient Model Selection for Sparse Hard and Fuzzy Center-Based Clustering

Efficient selection of the degree of sparsity for Sparse $k$-Means and Sparse Fuzzy $c$-Means is possible using expressions of Bayesian Information Criterion.

The derived expression for $k$-Means:

![skm_bic](https://raw.githubusercontent.com/Avisek20/sparse_clustering_BIC/master/imgs/skm_bic.png)

The derived expression for Fuzzy $c$-Means:

![sfcm_bic](https://raw.githubusercontent.com/Avisek20/sparse_clustering_BIC/master/imgs/sfcm_bic.png)

In comparison to the tradition Parition Method (PM) that uses the GAP statistic, the use of the derived expressions of BIC lead to significant reductions in computation complexity, and therefore execution times:

![timings](https://raw.githubusercontent.com/Avisek20/sparse_clustering_BIC/master/imgs/timings.png)

**sparse_kmeans_BIC.py** contains the implementation of using BIC for the model selection of Sparse $k$-Means.

**sparse_fuzzy_cmeans_BIC.py** contains the implementation of using BIC for the model selection of Sparse Fuzzy $c$-Means.

**indices/** contain contending cluster validity indices.

**exp_timings/** contains the data generated for the experiment comparing execution times.
