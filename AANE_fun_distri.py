def AANE_fun(Net, Attri, d, *varargs):
    """Jointly embed Net and Attri into embedding representation H
       H = AANE_fun(Net,Attri,d)
       H = AANE_fun(Net,Attri,d,lambd,rho)
       H = AANE_fun(Net,Attri,d,lambd,rho,'Att')
       H = AANE_fun(Net,Attri,d,lambd,rho,'Att',splitnum, worknum)
    :param Net: the weighted adjacency matrix
    :param Attri: the attribute information matrix with row denotes nodes
    :param d: the dimension of the embedding representation
    :param lambd: the regularization parameter
    :param rho: the penalty parameter
    :param 'Att': refers to conduct Initialization from the SVD of Attri
    :param splitnum: the number of pieces we split the SA for limited cache
    :param worknum: the number of worker
    :return: the embedding representation H
    Copyright 2017, Xiao Huang and Jundong Li.
    $Revision: 1.0.0 $  $Date: 2017/11/06 00:00:00 $
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import svds
    from math import ceil
    import multiprocessing as mp
    '''################# Update functions #################'''
    def updateH():
        xtx = np.dot(Z.transpose(), Z) * 2 + rho * np.eye(d)
        output = mp.Manager().dict()
        # define the worker
        def workerH(blocki, output):
            hlist = []
            for spliti in range(int(splitnum / worknum)):
                indexblock = block * (int(splitnum / worknum) * blocki + spliti)  # Index for splitting
                sums = (Attri[:, range(indexblock, indexblock + min(n - indexblock, block))].transpose() * Attri).dot(Z) * 2
                for i in range(indexblock, indexblock + min(n - indexblock, block)):
                    neighbor = Z[nexidx[i], :]  # the set of adjacent nodes of node i
                    for j in range(1):
                        normi_j = np.linalg.norm(neighbor - H[i, :], axis=1)  # norm of h_i^k-z_j^k
                        nzidx = normi_j != 0  # Non-equal Index
                        if np.any(nzidx):
                            normi_j = (lambd * Net[i][nzidx]) / normi_j[nzidx]
                            hi = np.linalg.solve(xtx + normi_j.sum() * np.eye(d), sums[i - indexblock, :] + (
                                neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + rho * (
                                                          Z[i, :] - U[i, :]))
                        else:
                            hi = np.linalg.solve(xtx, sums[i - indexblock, :] + rho * (
                                Z[i, :] - U[i, :]))
                    hlist.extend(hi)
            output[blocki] = hlist
        processes = [mp.Process(target=workerH, args=(blocki, output)) for blocki in range(worknum)]
        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()
        hlist = []
        for i in range(worknum):
            hlist = hlist + output[i]
        return np.reshape(hlist, (n, d))
    def updateZ():
        xtx = np.dot(H.transpose(), H) * 2 + rho * np.eye(d)
        output = mp.Manager().dict()
        # define the worker
        def workerZ(blocki, output):
            zlist = []
            for spliti in range(int(splitnum / worknum)):
                indexblock = block * (int(splitnum / worknum) * blocki + spliti)  # Index for splitting
                sums = (Attri[:, range(indexblock, indexblock + min(n - indexblock, block))].transpose() * Attri).dot(H) * 2
                for i in range(indexblock, indexblock + min(n - indexblock, block)):
                    neighbor = H[nexidx[i], :]  # the set of adjacent nodes of node i
                    for j in range(1):
                        normi_j = np.linalg.norm(neighbor - Z[i, :], axis=1)  # norm of h_i^k-z_j^k
                        nzidx = normi_j != 0  # Non-equal Index
                        if np.any(nzidx):
                            normi_j = (lambd * Net[i][nzidx]) / normi_j[nzidx]
                            zi = np.linalg.solve(xtx + normi_j.sum() * np.eye(d), sums[i - indexblock, :] + (
                                neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + rho * (
                                                          H[i, :] + U[i, :]))
                        else:
                            zi = np.linalg.solve(xtx, sums[i - indexblock, :] + rho * (
                                H[i, :] + U[i, :]))
                    zlist.extend(zi)
            output[blocki] = zlist
        processes = [mp.Process(target=workerZ, args=(blocki, output)) for blocki in range(worknum)]
        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()
        zlist = []
        for i in range(worknum):
            zlist = zlist + output[i]
        return np.reshape(zlist, (n, d))

    '''################# Parameters #################'''
    maxiter = 2  # Max num of iteration
    [n, m] = Attri.shape  # n = Total num of nodes, m = attribute category num
    Net = sparse.lil_matrix(Net)
    Net.setdiag(np.zeros(n))
    Net = csc_matrix(Net)
    Attri = csc_matrix(Attri)
    lambd = 0.1  # Initial regularization parameter
    rho = 5  # Initial penalty parameter
    splitnum = 60  # number of pieces we split the SA for limited cache
    worknum = 3  # number of worker used for distribution
    if len(varargs) > 0:
        lambd = varargs[0]
        rho = varargs[1]
        if len(varargs) >=3 and varargs[2] == 'Att':
            sumcol = Attri.sum(0)
            H = svds(Attri[:, sorted(range(m), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, m)]], d)[0]
        else:
            sumcol = Net.sum(0)
            H = svds(Net[:, sorted(range(n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, n)]], d)[0]
        if len(varargs) >=4:
            worknum = varargs[4]
            splitnum = ceil(float(varargs[3]/varargs[4])) * varargs[4]
    else:
        sumcol = Net.sum(0)
        H = svds(Net[:, sorted(range(n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10*d, n)]], d)[0]
    block = int(ceil(float(n) / splitnum))
    with np.errstate(divide='ignore'):  # inf will be ignored
        Attri = Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5)))
    Z = H.copy()
    U = np.zeros((n, d))
    nexidx = np.split(Net.indices, Net.indptr[1:-1])
    Net = np.split(Net.data, Net.indptr[1:-1])
    '''################# Iterations #################'''
    H = updateH()
    for __ in range(maxiter - 1):
        Z = updateZ()
        U = U + H - Z
        H = updateH()
    return H