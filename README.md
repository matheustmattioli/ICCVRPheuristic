# IC_VRPheuristic
Heuristics and metaheuristics for capacitated vehicle routing problem (CVRP) using python implementing the following algorithms:
  - Greedy Randomized Adaptative Search Procedure (GRASP)
  - Variable Neighbourhood Descent (VND) to solve Travelling Salesman Problems, TSPs, inside VRP.
  - In VND we use Local Search with 2OPT and 3OPT neighbour structure.
  - Clustering algorithms.
  
Compile the program using `python solve.py [instance] [clustering_method]`.
Where `clustering_method` can be:
- `1` for greedypath clustering
- `2` for serial clustering
- `3` for parallel clustering

