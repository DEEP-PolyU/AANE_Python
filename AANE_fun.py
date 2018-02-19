def AANE_fun(Net, Attri, d, *varargs):
    """Jointly embed Net and Attri into embedding representation H
       H = AANE_fun(Net,Attri,d)
       H = AANE_fun(Net,Attri,d,lambd,rho)
       H = AANE_fun(Net,Attri,d,lambd,rho,maxiter)
       H = AANE_fun(Net,Attri,d,lambd,rho,maxiter,'Att')
       H = AANE_fun(Net,Attri,d,lambd,rho,maxiter,'Att',splitnum)
    :param Net: the weighted adjacency matrix
    :param Attri: the attribute information matrix with row denotes nodes
    :param d: the dimension of the embedding representation
    :param lambd: the regularization parameter
    :param rho: the penalty parameter
    :param maxiter: the maximum number of iteration
    :param 'Att': refers to conduct Initialization from the SVD of Attri
    :param splitnum: the number of pieces we split the SA for limited cache
    :return: the embedding representation H
    Copyright 2017 & 2018, Xiao Huang and Jundong Li.
    $Revision: 1.0.2 $  $Date: 2018/02/19 00:00:00 $
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import svds
    from math import ceil
    '''################# Parameters #################'''
    global affi, sa, H, Z
    maxiter = 2  # Max num of iteration
    [n, m] = Attri.shape  # n = Total num of nodes, m = attribute category num
    Net = sparse.lil_matrix(Net)
    Net.setdiag(np.zeros(n))
    Net = csc_matrix(Net)
    Attri = csc_matrix(Attri)
    lambd = 0.05  # Initial regularization parameter
    rho = 5  # Initial penalty parameter
    splitnum = 1  # number of pieces we split the SA for limited cache
    if len(varargs) >= 4 and varargs[3] == 'Att':
        sumcol = Attri.sum(0)
        H = svds(Attri[:, sorted(range(m), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, m)]], d)[0]
    else:
        sumcol = Net.sum(0)
        H = svds(Net[:, sorted(range(n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, n)]], d)[0]

    if len(varargs) > 0:
        lambd = varargs[0]
        rho = varargs[1]
        if len(varargs) >= 3:
            maxiter = varargs[2]
            if len(varargs) >=5:
                splitnum = varargs[4]
    block = min(int(ceil(float(n) / splitnum)), 7575)  # Treat at least each 7575 nodes as a block
    splitnum = int(ceil(float(n) / block))
    with np.errstate(divide='ignore'):  # inf will be ignored
        Attri = Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5)))
    Z = H.copy()
    affi = -1  # Index for affinity matrix sa
    U = np.zeros((n, d))
    nexidx = np.split(Net.indices, Net.indptr[1:-1])
    Net = np.split(Net.data, Net.indptr[1:-1])
    '''################# Update functions #################'''
    def updateH():
        global affi, sa, H
        xtx = np.dot(Z.transpose(), Z) * 2 + rho * np.eye(d)
        for blocki in range(splitnum):  # Split nodes into different Blocks
            indexblock = block * blocki  # Index for splitting blocks
            if affi != blocki:
                sa = Attri[:, range(indexblock, indexblock + min(n - indexblock, block))].transpose() * Attri
                affi = blocki
            sums = sa.dot(Z) * 2
            for i in range(indexblock, indexblock + min(n - indexblock, block)):
                neighbor = Z[nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - H[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (lambd * Net[i][nzidx]) / normi_j[nzidx]
                        H[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(d), sums[i - indexblock, :] + (
                            neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + rho * (
                                                      Z[i, :] - U[i, :]))
                    else:
                        H[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + rho * (
                            Z[i, :] - U[i, :]))
    def updateZ():
        global affi, sa, Z
        xtx = np.dot(H.transpose(), H) * 2 + rho * np.eye(d)
        for blocki in range(splitnum):  # Split nodes into different Blocks
            indexblock = block * blocki  # Index for splitting blocks
            if affi != blocki:
                sa = Attri[:, range(indexblock, indexblock + min(n - indexblock, block))].transpose() * Attri
                affi = blocki
            sums = sa.dot(H) * 2
            for i in range(indexblock, indexblock + min(n - indexblock, block)):
                neighbor = H[nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - Z[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (lambd * Net[i][nzidx]) / normi_j[nzidx]
                        Z[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(d), sums[i - indexblock, :] + (
                            neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + rho * (
                                                      H[i, :] + U[i, :]))
                    else:
                        Z[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + rho * (
                            H[i, :] + U[i, :]))
    '''################# First update H #################'''
    updateH()
    '''################# Iterations #################'''
    for iternum in range(maxiter - 1):
        updateZ()
        U = U + H - Z
        updateH()
    return H
