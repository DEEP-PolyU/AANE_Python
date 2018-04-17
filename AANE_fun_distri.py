import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from numpy.linalg import solve
from math import ceil
import multiprocessing as mp


class AANE:
    """Jointly embed Net and Attri into embedding representation H
    H = AANE_fun(Net,Attri,d)
    H = AANE_fun(Net,Attri,d,lambd,rho)
    H = AANE_fun(Net,Attri,d,lambd,rho,maxiter)
    H = AANE_fun(Net,Attri,d,lambd,rho,maxiter,'Att')
    H = AANE_fun(Net,Attri,d,lambd,rho,maxiter,'Att', worknum)
    H = AANE_fun(Net,Attri,d,lambd,rho,maxiter,'Att', worknum, splitnum)
    :param Net: the weighted adjacency matrix
    :param Attri: the attribute information matrix with row denotes nodes
    :param d: the dimension of the embedding representation
    :param lambd: the regularization parameter
    :param rho: the penalty parameter
    :param maxiter: the maximum number of iteration
    :param 'Att': refers to conduct Initialization from the SVD of Attri
    :param worknum: the number of worker
    :param splitnum: number of pieces we split the SA for limited cache
    :return: the embedding representation H
    Copyright 2017 & 2018, Xiao Huang and Jundong Li.
    $Revision: 1.0.3 $  $Date: 2018/04/05 00:00:00 $
    """
    def __init__(self, Net, Attri, d, *varargs):
        # shared memory
        #self.output = mp.Manager().dict()

        #Share by workers
        #self.blocki = None
        #self.block = None
        #self.d = None
        #self.lambd = None
        #self.rho = None
        #self.Net = None
        #self.Attri = None
        #self.Z = None
        #self.H = None
        #self.nexidx = None
        #self.U = None
        #self.maxiter = None
        #self.n = None
        #self.m = None
        #self.worknum = None
        #self.splitnum = None

        self.maxiter = 2  # Max num of iteration
        [self.n, m] = Attri.shape  # n = Total num of nodes, m = attribute category num
        Net = sparse.lil_matrix(Net)
        Net.setdiag(np.zeros(self.n))
        Net = csc_matrix(Net)
        Attri = csc_matrix(Attri)
        self.lambd = 0.05  # Initial regularization parameter
        self.rho = 5  # Initial penalty parameter
        self.worknum = 3  # number of worker used for distribution
        self.splitnum = self.worknum  # number of pieces we split the SA for limited cache

        if len(varargs) >= 4 and varargs[3] == 'Att':
            sumcol = np.arange(m)
            np.random.shuffle(sumcol)
            self.H = svds(Attri[:, sumcol[0:min(10 * d, m)]], d)[0]
        else:
            sumcol = Net.sum(0)
            self.H = svds(Net[:, sorted(range(self.n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, self.n)]], d)[0]
        if len(varargs) > 0:
            self.lambd = varargs[0]
            self.rho = varargs[1]
            if len(varargs) >= 3:
                self.maxiter = varargs[2]
                if len(varargs) >= 5:
                    self.worknum = int(varargs[4])
                    self.splitnum = self.worknum
                    if len(varargs) >= 6:
                        self.splitnum = int(ceil(float(varargs[5] / self.worknum)) * self.worknum)
                self.block = int(ceil(float(self.n) / self.splitnum))
        with np.errstate(divide='ignore'):  # inf will be ignored
            self.Attri = Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5)))

        self.Z = self.H.copy()
        self.U = np.zeros((self.n, d))
        self.nexidx = np.split(Net.indices, Net.indptr[1:-1])
        self.Net = np.split(Net.data, Net.indptr[1:-1])
        self.d = d

    def funtion(self):
        self.H = self.updateH()
        for __ in range(self.maxiter - 1):
            self.Z = self.updateZ()
            self.U = self.U + self.H - self.Z
            self.H = self.updateH()
        return self.H

    def updateH(self):
        output = mp.Manager().dict()
        xtx = np.dot(self.Z.transpose(), self.Z) * 2 + self.rho * np.eye(self.d)
        with mp.Pool(processes=self.worknum) as pool:
            result = pool.map_async(self.workerH, ((blocki, xtx, output) for blocki in range(self.splitnum)))
            #result.get(timeout=self.worknum)
            result.get()
            pool.terminate()
        hlist = []
        for i in range(self.splitnum):
            hlist = hlist + output[i]
        return np.reshape(hlist, (self.n, self.d))

    def updateZ(self):
        output = mp.Manager().dict()
        xtx = np.dot(self.H.transpose(), self.H) * 2 + self.rho * np.eye(self.d)
        with mp.Pool(processes=self.worknum) as pool:
            result = pool.map_async(self.workerZ, ((blocki, xtx, output) for blocki in range(self.splitnum)))
            result.get()
            pool.terminate()
        zlist = []
        for i in range(self.splitnum):
            zlist = zlist + output[i]
        return np.reshape(zlist, (self.n, self.d))

    def workerH(self, tup):
        blocki, xtx, output = tup
        hlist = []
        indexblock = self.block * blocki
        sums = (self.Attri[:, indexblock: indexblock + min(self.n - indexblock, self.block)].transpose() * self.Attri).dot(self.Z) * 2
        for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
            neighbor = self.Z[self.nexidx[i], :]  # the set of adjacent nodes of node i
            normi_j = np.linalg.norm(neighbor - self.H[i, :], axis=1)  # norm of h_i^k-z_j^k
            nzidx = normi_j != 0  # Non-equal Index
            if np.any(nzidx):
                normi_j = (self.lambd * self.Net[i][nzidx]) / normi_j[nzidx]
                hi = solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - indexblock, :] + (
                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (
                    self.Z[i, :] - self.U[i, :]))
            else:
                hi = solve(xtx, sums[i - indexblock, :] + self.rho * (self.Z[i, :] - self.U[i, :]))
            hlist.extend(hi)
        output[blocki] = hlist

    def workerZ(self, tup):
        blocki, xtx, output = tup
        zlist = []
        indexblock = self.block * blocki
        sums = (self.Attri[:, indexblock: indexblock + min(self.n - indexblock, self.block)].transpose() * self.Attri).dot(self.H) * 2
        for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
            neighbor = self.H[self.nexidx[i], :]  # the set of adjacent nodes of node i
            normi_j = np.linalg.norm(neighbor - self.Z[i, :], axis=1)  # norm of h_i^k-z_j^k
            nzidx = normi_j != 0  # Non-equal Index
            if np.any(nzidx):
                normi_j = (self.lambd * self.Net[i][nzidx]) / normi_j[nzidx]
                zi = solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - indexblock, :] + (
                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (
                    self.H[i, :] + self.U[i, :]))
            else:
                zi = solve(xtx, sums[i - indexblock, :] + self.rho * (self.H[i, :] + self.U[i, :]))
            zlist.extend(zi)
        output[blocki] = zlist

