import numpy as np
import scipy.io as sio
from AANE_fun_distri import AANE
import time

print("Current version for distributed computing could only be run on macOS.")

'''################# Load data  #################'''
mat_contents = sio.loadmat('BlogCatalog.mat')
lambd = 10**-0.6  # the regularization parameter
rho = 5  # the penalty parameter

# mat_contents = sio.loadmat('Flickr.mat')
# lambd = 0.0425  # the regularization parameter
# rho = 4  # the penalty parameter

'''################# Experimental Settings #################'''
d = 100  # the dimension of the embedding representation
maxiter = 2  # the maximum number of iteration
G = mat_contents["Network"]
A = mat_contents["Attributes"]
del mat_contents
n = G.shape[0]
Indices = np.random.randint(25, size=n)  # 5-fold cross-validation indices

Group1 = []
Group2 = []
[Group1.append(x) for x in range(0, len(Indices)) if Indices[x] <= 20]  # 2 for 10%, 5 for 25%, 20 for 100% of training group
[Group2.append(x) for x in range(0, len(Indices)) if Indices[x] >= 21]  # test group
n1 = len(Group1)  # num of nodes in training group
n2 = len(Group2)  # num of nodes in test group
CombG = G[Group1+Group2, :][:, Group1+Group2]
CombA = A[Group1+Group2, :]

'''################# Accelerated Attributed Network Embedding #################'''
if __name__ == "__main__":
    print("Accelerated Attributed Network Embedding (AANE), 5-fold with 100% of training is used:")
    start_time = time.time()
    H = AANE(CombG, CombA, d, lambd, rho, maxiter, 'Net', 1, 6).funtion()  #  worknum=1, splitnum=6
    print("time elapsed with 1 worker: {:.2f}s".format(time.time() - start_time))
    sio.savemat('H.mat', {"H": H})
    start_time = time.time()
    H = AANE(CombG, CombA, d, lambd, rho, maxiter, 'Net', 2).funtion()
    print("time elapsed with 2 workers: {:.2f}s".format(time.time() - start_time))
    start_time = time.time()
    H = AANE(CombG, CombA, d, lambd, rho, maxiter, 'Net', 3).funtion()
    print("time elapsed with 3 workers: {:.2f}s".format(time.time() - start_time))