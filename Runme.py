import numpy as np
import scipy.io as sio
from aane_fun import aane_fun
import time


'''################# Load data  #################'''
mat_contents = sio.loadmat('BlogCatalog.mat')
lambd = 10**-0.6  # the regularization parameter
rho = 5  # the penalty parameter

# mat_contents = sio.loadmat('Flickr.mat')
# lambd = 0.0425  # the regularization parameter
# rho = 4  # the penalty parameter

'''################# Experimental Settings #################'''
d = 100  # the dimension of the embedding representation
G = mat_contents["Network"]
A = mat_contents["Attributes"]
Label = mat_contents["Label"]
del mat_contents
n = G.shape[0]
Indices = np.random.randint(25, size=n)  # 5-fold cross-validation indices

Group1 = []
Group2 = []
[Group1.append(x) for x in range(0, n) if Indices[x] <= 10]  # 2 for 10%, 5 for 25%, 20 for 100% of training group
[Group2.append(x) for x in range(0, n) if Indices[x] >= 21]  # test group
n1 = len(Group1)  # num of nodes in training group
n2 = len(Group2)  # num of nodes in test group
CombG = G[Group1+Group2, :][:, Group1+Group2]
CombA = A[Group1+Group2, :]

'''################# Accelerated Attributed Network Embedding #################'''
print 'Accelerated Attributed Network Embedding (AANE), 5-fold with 50% of training is used:'
start_time = time.time()
h1 = aane_fun(CombG, CombA, d, lambd, rho)
print "time elapsed: {:.2f}s".format(time.time() - start_time)

'''################# AANE for a Pure Network #################'''
print 'AANE for a pure network:'
start_time = time.time()
h2 = aane_fun(CombG, CombG, d, lambd, rho)
print "time elapsed: {:.2f}s".format(time.time() - start_time)
sio.savemat('Embedding.mat', {"H_AANE": h1}, {"H_net": h2})