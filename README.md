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
- Output: Embedding.mat, with "H_AANE" denotes the attributed network embedding, and "H_net" denotes the network embedding

## Code in Python
```
h = aane_fun(net,attri,d);  
h = aane_fun(net,attri,d,lambda,rho);  
h = aane_fun(net,attri,d,lambda,rho,'Att');  
h = aane_fun(net,attri,d,lambda,rho,'Att',worknum);  
```

- h is the joint embedding representation of net and attri;
- net is the weighted adjacency matrix;
- attri is the node attribute information matrix with row denotes nodes.


## Reference in BibTeX: 
@conference{Huang-etal17Accelerated,  
Author = {Xiao Huang and Jundong Li and Xia Hu},  
Booktitle = {SIAM International Conference on Data Mining},  
Pages = {633--641},  
Title = {Accelerated Attributed Network Embedding},  
Year = {2017}}
