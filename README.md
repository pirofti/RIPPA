# Restarted Inexact Proximal Point Algorithm (RIPPA)

Implementation of the RIPPA algorithm for convex optimization described in
A. Pătrașcu and P. Irofti, "Computational complexity of Inexact Proximal
Point Algorithms forConvex Optimization"

If you use our [work](https://arxiv.org/pdf/2108.04482) in your research, please cite as:
```
@article{PI21,
  title={Computational complexity of Inexact Proximal Point Algorithm for Convex Optimization under Holderian Growth}, 
  author = {Pătrașcu, A. and Irofti, P.},
  year={2021},
  eprint={2108.04482},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
}
```

The algorithm is implemented in [rippa.py](rippa.py). Have a look at the applications for full examples:
* [Robust L1 Least Squares](test_rippa_robust_l1ls.py) 
* [Graph Support Vector Machines](test_rippa_graphsvm.py) 
* [Matrix Completion](test_rippa_matrixcompletion.py) 
