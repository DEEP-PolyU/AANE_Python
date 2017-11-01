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
H = AANE_fun(Net,Attri,d)
H = AANE_fun(Net,Attri,d,lambd,rho)
H = AANE_fun(Net,Attri,d,lambd,rho,'Att')
H = AANE_fun(Net,Attri,d,lambd,rho,'Att',splitnum)
```

- H is the joint embedding representation of Net and Attri;
- Net is the weighted adjacency matrix;
- Attri is the node attribute information matrix with row denotes nodes.


## Reference in BibTeX: 
@conference{Huang-etal17Accelerated,  
Author = {Xiao Huang and Jundong Li and Xia Hu},  
Booktitle = {SIAM International Conference on Data Mining},  
Pages = {633--641},  
Title = {Accelerated Attributed Network Embedding},  
Year = {2017}}
