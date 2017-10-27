# AANE_Python
Accelerated Attributed Network Embedding, SDM 2017

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

## Requirements
- numpy
- scipy

## Reference in BibTeX: 
@conference{Huang-etal17Accelerated,  
Author = {Xiao Huang and Jundong Li and Xia Hu},  
Booktitle = {SIAM International Conference on Data Mining},  
Pages = {633--641},  
Title = {Accelerated Attributed Network Embedding},  
Year = {2017}}
