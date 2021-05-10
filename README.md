# IC_VRPheuristic
Heuristics and metaheuristics for capacitated vehicle routing problem (CVRP) using python implementing these algorithms:
  - Greedy Randomized Adaptative Search Procedure (GRASP)
  - Variable Neighbourhood Search (VNS) to solve Travelling Salesman Problems, TSPs, inside VRP.
  - In VNS we use Local Search with 2OPT and 3OPT neighbour structure.
  - Clustering algorithms.
 
In next updates we want to implement a Local Search for VRP. 
 
Compile the program using `python solve.py [instance] [clustering_method]`.
Where `clustering_method` can be:
- `1` for greedypath clustering
- `2` for serial clustering
- `3` for parallel clustering

