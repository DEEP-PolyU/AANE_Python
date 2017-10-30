def aane_fun(net, attri, d, *varargs):
    """Jointly embed net and attri into embedding representation h
       h = aane_fun(net,attri,d);
       h = aane_fun(net,attri,d,lambda,rho);
       h = aane_fun(net,attri,d,lambda,rho,'Att');
       h = aane_fun(net,attri,d,lambda,rho,'Att',worknum);
    :param net: the weighted adjacency matrix
    :param attri: the attribute information matrix with row denotes nodes
    :param d: the dimension of the embedding representation
    :param lambd: the regularization parameter
    :param rho: the penalty parameter
    :param 'Att': refers to conduct Initialization from the SVD of Attri
    :param splitnum: the number of pieces we split the SA for limited cache
    :return: the embedding representation h
    Copyright 2017, Xiao Huang and Jundong Li.
    $Revision: 1.0.0 $  $Date: 2017/10/25 00:00:00 $
    """
    global affi, sa, h, z
    import numpy as np
    from scipy import sparse
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import svds
    from math import ceil

    '''################# Parameters #################'''
    maxiter = 2  # Max num of iteration
    [n, m] = attri.shape  # n = Total num of nodes, m = attribute category num
    net = sparse.lil_matrix(net)
    net.setdiag(np.zeros(n))
    net = csc_matrix(net)
    attri = csc_matrix(attri)
    lambd = 0.1  # Initial regularization parameter
    rho = 5  # Initial penalty parameter
    splitnum = 1  # number of pieces we split the SA for limited cache
    if len(varargs) > 0:
        lambd = varargs[0]
        rho = varargs[1]
        if len(varargs) >=3 and varargs[2] == 'Att':
            sumcol = attri.sum(0)
            h = svds(attri[:, sorted(range(m), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, m)]], d)[0]
        else:
            sumcol = net.sum(0)
            h = svds(net[:, sorted(range(n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, n)]], d)[0]
        if len(varargs) >=4:
            splitnum = varargs[3]
    else:
        sumcol = net.sum(0)
        h = svds(net[:, sorted(range(n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10*d, n)]], d)[0]
    block = min(int(ceil(float(n) / splitnum)), 7575)  # Treat at least each 7575 nodes as a block
    splitnum = int(ceil(float(n) / block))
    with np.errstate(divide='ignore'):  # inf will be ignored
        attri = attri.transpose() * sparse.diags(np.ravel(np.power(attri.power(2).sum(1), -0.5)))
    z = h.copy()
    affi = -1  # Index for affinity matrix sa
    u = np.zeros((n, d))
    nexidx = np.split(net.indices, net.indptr[1:-1])
    net = np.split(net.data, net.indptr[1:-1])
    '''################# Update functions #################'''
    def updateh():
        global affi, sa, h
        xtx = np.dot(z.transpose(), z) * 2 + rho * np.eye(d)
        for blocki in range(splitnum):  # Split nodes into different Blocks
            indexblock = block * blocki  # Index for splitting blocks
            if affi != blocki:
                sa = attri[:, range(indexblock, indexblock + min(n - indexblock, block))].transpose() * attri
                affi = blocki
            sums = sa.dot(z) * 2
            for i in range(indexblock, indexblock + min(n - indexblock, block)):
                neighbor = z[nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - h[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (lambd * net[i][nzidx]) / normi_j[nzidx]
                        h[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(d), sums[i - indexblock, :] + (
                            neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + rho * (
                                                      z[i, :] - u[i, :]))
                    else:
                        h[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + rho * (
                            z[i, :] - u[i, :]))
    def updatez():
        global affi, sa, z
        xtx = np.dot(h.transpose(), h) * 2 + rho * np.eye(d)
        for blocki in range(splitnum):  # Split nodes into different Blocks
            indexblock = block * blocki  # Index for splitting blocks
            if affi != blocki:
                sa = attri[:, range(indexblock, indexblock + min(n - indexblock, block))].transpose() * attri
                affi = blocki
            sums = sa.dot(h) * 2
            for i in range(indexblock, indexblock + min(n - indexblock, block)):
                neighbor = h[nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - z[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (lambd * net[i][nzidx]) / normi_j[nzidx]
                        z[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(d), sums[i - indexblock, :] + (
                            neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + rho * (
                                                      h[i, :] + u[i, :]))
                    else:
                        z[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + rho * (
                            h[i, :] + u[i, :]))
    '''################# First update h #################'''
    updateh()
    '''################# Iterations #################'''
    for iternum in range(maxiter - 1):
        updatez()
        u = u + h - z
        updateh()
    return h
