import numpy as np

from numpy.random import default_rng
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import cholesky
from itertools import chain


def er_dag(p, d=0.5, ad=None, rng=default_rng()):
    '''
    Randomly generates an Erdos-Renyi (lower triangular) direct acyclic graph
    given an ordering.

    Parameters
    ----------
    p = |variables|
    d = |edges| / |possible edges|   (ignored if ad is not None)
    ad = average degree
    rng = random number generator

    Returns
    -------
    g = direct acyclic graph
    '''

    # npe = |possible edges|
    npe = int(p * (p - 1) / 2)

    # ne = |edges|
    if ad is not None: d = ad / (p - 1)
    ne = int(d * npe)

    # generate edges
    e = np.append(np.zeros(npe - ne, np.uint8), np.ones(ne, np.uint8))
    rng.shuffle(e)

    # generate graph
    g = np.zeros([p, p], np.uint8)
    g.T[np.triu_indices(p, 1)] = e

    return g


def sf_out(g, rng=default_rng()):
    '''
    Rewires entries within rows-row sum (in-degree) doesnt change.

    Parameters
    ----------
    g = directed acyclic graph
    rng = random number generator

    Returns
    -------
    g = direct acyclic graph
    '''

    # p = |variables|
    p = g.shape[0]

    # reorder g if not lower triangular
    nlt = any([g[i, j] for j in range(p) for i in range(j)])
    if nlt:
        ord = sofic_order(g)
        g = g[ord][:, ord]
    else:
        g = g.copy()

    for i in range(1, p):
        J = [[j] for j in range(i)]
        J += [[j] * int(np.sum(g[:i, j])) for j in range(i)]
        J = list(chain.from_iterable(J))
        rng.shuffle(J)

        in_deg = np.sum(g[i])
        g[i] = np.zeros(p)

        for j in J:
            if in_deg == 0: break
            if g[i, j] == 0:
                in_deg -= 1
                g[i, j] = 1

    # reorder g if not lower triangular
    if nlt:
        ord = invert_order(ord)
        g = g[ord][:, ord]

    return g


def sf_in(g, rng=default_rng()):
    '''
    Rewires entries within cols-col sum (out-degree) doesnt change.

    Parameters
    ----------
    g = directed acyclic graph
    rng = random number generator

    Returns
    -------
    g = direct acyclic graph
    '''

    # p = |variables|
    p = g.shape[0]

    # reorder g if not lower triangular
    nlt = any([g[i, j] for j in range(p) for i in range(j)])
    if nlt:
        ord = sofic_order(g)
        g = g[ord][:, ord]
    else:
        g = g.copy()

    for i in range(1, p):
        J = [[j] for j in range(i)]
        J += [[j] * int(np.sum(g[p - j - 1, p - i:])) for j in range(i)]
        J = list(chain.from_iterable(J))
        rng.shuffle(J)

        out_deg = np.sum(g[:, p - i - 1])
        g[:, p - i - 1] = np.zeros(p)

        for j in J:
            if out_deg == 0: break
            if g[p - j - 1, p - i - 1] == 0:
                out_deg -= 1
                g[p - j - 1, p - i - 1] = 1

    # reorder g if not lower triangular
    if nlt:
        ord = invert_order(ord)
        g = g[ord][:, ord]

    return g


def num_source(g):
    '''
    Helper function: counts the number of source variables.

    Parameters
    -----------
    g = directed acyclic graph

    Returns:
    --------
    m = source count
    '''

    # p = |variables|
    p = g.shape[0]

    src = [i for i in range(p) if np.sum(g[i]) == 0]

    return len(src)


def sofic_order(g):
    '''
    Helper function: returns a source first consistent order.

    Parameters
    -----------
    g = directed acyclic graph

    Returns:
    --------
    ord = order
    '''

    # p = |variables|
    p = g.shape[0]

    # convert g to booleans
    g = g.astype(bool)

    ord = [i for i in range(p) if np.sum(g[i]) == 0]

    while len(ord) < p:
        for i in range(p):
            if i in ord: continue
            if np.sum(g[i]) == np.sum(g[i][ord]):
                ord.append(i)
                break
            if i == p - 1:
                raise ValueError("cycle detected")

    return ord


def invert_order(ord):
    '''
    Helper function: inverts the order.

    Parameters
    -----------
    ord = order

    Returns:
    --------
    inv_ord = inverse order
    '''

    # p = |variables|
    p = len(ord)

    inv_ord = [0 for i in range(p)]
    for i in range(p): inv_ord[ord[i]] = i

    return inv_ord


def mpii(g, i, rng=default_rng()):
    '''
    Helper function: samples a multivariate Pearson type II.

    Parameters
    -----------
    g = directed acyclic graph
    i = index
    rng = random number generator

    Returns:
    --------
    w = mpii sample
    '''

    # p = |variables|
    p = g.shape[0]

    # k = |parents|
    k = np.sum(g[i])

    # initialize w
    w = np.zeros(i)

    # update w
    if k > 0:

        q = rng.beta(k / 2, (p - i) / 2)
        y = rng.standard_normal(k)
        u = y / norm(y)
        w[:k] = np.sqrt(q) * u

    return w


def pmat(g, i):
    '''
    Helper function: returns a permutation matrix.

    Parameters:
    -----------
    g = directed acyclic graph
    i = index

    Returns:
    --------
    P = permutation matrix
    '''

    P = np.zeros([i, i], dtype=np.uint8)
    k = 0

    for q in (1, 0):
        for j in np.where(g[i, :i] == q)[0]:
            P[j, k] = 1
            k += 1

    return P


def corr(g):
    '''
    Randomly generates a correlation matrix where f(R) ~ 1 given a direct
    acyclic graph.

    Parameters
    ----------
    g = directed acyclic graph

    Returns
    -------
    R = correlation matrix
    B = beta matrix
    O = error vector
    '''

    # reorder g
    ord = sofic_order(g)
    g = g[ord][:, ord]

    # p = |variables|; m = |source variables|
    p = g.shape[0]
    m = num_source(g)

    # initialize correlation / coefficient / error matrices
    R = np.eye(p)
    B = np.zeros([p, p])
    O = np.ones(p)

    for i in range(m, p):
        P = pmat(g, i)
        L = cholesky(P.T @ R[:i, :i] @ P)
        w = mpii(g, i)

        r = P @ L @ w
        b = P @ inv(L).T @ w
        o = 1 - np.sum(w * w)

        R[:i, i] = r
        R[i, :i] = r
        B[i, :i] = b
        O[i] = o

    # reorder R, B, and O
    ord = invert_order(ord)
    R = R[ord][:, ord]
    B = B[ord][:, ord]
    O = O[ord]

    return R, B, O


def cov(g, lb_b=0, ub_b=1, lb_o=1, ub_o=2, rng=default_rng()):
    '''
    Randomly generates a covariance matrix given a directed acyclic graph.

    Parameters
    ----------
    g = directed acyclic graph
    lb_b = lower bound for beta
    ub_b = upper bound for beta
    lb_o = lower bound for omega
    ub_o = upper bound for omega
    rng = random number generator

    Returns
    -------
    S = covariance matrix
    B = beta matrix
    O = error vector
    '''

    # p = |variables|
    p = g.shape[0]

    # reorder g
    ord = sofic_order(g)
    g = g[ord][:, ord]

    # e = |edges|
    e = np.sum(g)

    # generate edge weights
    B = np.zeros([p, p])
    B[np.where(g)] = rng.choice([-1, 1], e) * rng.uniform(lb_b, ub_b, e)

    # generate variance terms
    O = rng.uniform(lb_o, ub_o, p)

    # calculate covariance
    IB = inv(np.eye(p) - B)
    S = IB @ np.diag(O) @ IB.T

    # reorder S, B, and O
    ord = invert_order(ord)
    S = S[ord][:, ord]
    B = B[ord][:, ord]
    O = O[ord]

    return S, B, O


def simulate(B, O, n, err=None, rng=default_rng()):
    '''
    Randomly simulates data with the provided parameters.

    Parameters
    ----------
    B = beta matrix
    O = error vector
    n = sample size
    err = error distribution
    rng = random number generator

    Returns
    -------
    X = data
    '''

    # p = |variables|
    p = B.shape[0]

    # reorder B and O
    ord = sofic_order(B)
    B = B[ord][:, ord]
    O = O[ord]

    # set default error as normal
    if err is None: err = lambda *x: rng.normal(0, np.sqrt(x[0]), x[1])

    # simulate data
    X = np.zeros([n, p])
    for i in range(p):
        # parents
        J = np.where(B[i])[0]

        # linear effect
        for j in J: X[:, i] += B[i, j] * X[:, j]

        # add error
        X[:, i] += err(O[i], n)

    # reorder X
    ord = invert_order(ord)
    X = X[:, ord]

    return X


def standardize(X):
    '''
    Standardizes the data.

    Parameters
    ----------
    X = data

    Returns
    -------
    X = data
    '''

    return (X - X.mean(0)) / X.std(0)


def randomize_graph(g, rng=default_rng()):
    '''
    Randomly reorders the variables of the graph.

    Parameters
    ----------
    g = directed acyclic graph
    rng = random number generator

    Returns
    -------
    g = directed acyclic graph
    '''

    # p = |variables|
    p = g.shape[0]

    # random reorder g
    pi = [i for i in range(p)]
    rng.shuffle(pi)

    return g[pi][:, pi]


def cov_to_corr(S):
    '''
    Rescales covariance to correlation.

    Parameters
    ----------
    S = covariance matrix

    Returns
    -------
    R = correlation matrix
    '''

    D = np.diag(np.sqrt(np.diag(S)))
    ID = inv(D)

    return ID @ S @ ID


def cov_to_dag(g, S):
    '''
    Converts covariance to directed acyclic graph parameters.

    Parameters
    ----------
    g = directed acyclic graph
    S = covariance matrix

    Returns
    -------
    B = beta matrix
    O = error vector
    '''

    # p = |variables|
    p = S.shape[0]

    B = np.zeros((p, p))
    O = np.diag(S)

    for i in range(p):
        pa = np.where(g[i])[0]
        if len(pa) > 0:
            yX = S[i, pa]
            XX = S[np.ix_(pa, pa)]
            IXX = inv(XX)

            B[i, pa] = yX @ IXX
            O[i] -= yX @ IXX @ yX

    return B, O
