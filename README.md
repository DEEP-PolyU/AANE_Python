# AANE_Python
Accelerated Attributed Network Embedding, SDM 2017



## Installation
- Requirements
1. numpy
2. scipy
- Usage
1. cd AANE_Python
2. pip install -r requirements.txt
3. python Runme.py

## Input and Output
- Input: dataset such as "BlogCatalog.mat" and "Flickr.mat"
- Output: Embedding.mat, with "H_AANE" denotes the attributed network embedding, and "H_Net" denotes the network embedding

## Code in Python
```
from AANE_fun import AANE_fun
H = AANE_fun(Net,Attri,d)
H = AANE_fun(Net,Attri,d,lambd,rho)
H = AANE_fun(Net,Attri,d,lambd,rho,'Att')
H = AANE_fun(Net,Attri,d,lambd,rho,'Att',splitnum)
```

- H is the joint embedding representation of Net and Attri;
- Net is the weighted adjacency matrix;
- Attri is the node attribute information matrix with row denotes nodes;
- splitnum is the number of pieces we split the SA for limited cache.
- Python 3.6.3 or 2.7.13 is recommended.

## Reference in BibTeX: 
@conference{Huang-etal17Accelerated,  
Author = {Xiao Huang and Jundong Li and Xia Hu},  
Booktitle = {SIAM International Conference on Data Mining},  
Pages = {633--641},  
Title = {Accelerated Attributed Network Embedding},  
Year = {2017}}


## Code for Distributed Computing
```
from AANE_fun_distri import AANE_fun
H = AANE_fun(Net,Attri,d)
H = AANE_fun(Net,Attri,d,lambd,rho)
H = AANE_fun(Net,Attri,d,lambd,rho,'Att')
H = AANE_fun(Net,Attri,d,lambd,rho,'Att',splitnum, worknum)
```
- H is the joint embedding representation of Net and Attri;
- Net is the weighted adjacency matrix;
- Attri is the node attribute information matrix with row denotes nodes;
- splitnum is the number of pieces we split the SA for limited cache;
- worknum is the number of worker.

The function for distributed computing could only be run on macOS with Python 3.6.3 recommended.
