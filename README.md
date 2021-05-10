# IC_VRPheuristic
Heuristc and metaheuristics for capacitated vehicle routing problem (CVRP).
For now using python with those algorithms:
  - Greedy Randomized Adaptative Search Procedure (GRASP)
  - Variable Neighbourhood Search (VNS) to solve Travelling Salesman Problems, TSPs, inside VRP.
  - In VNS we use Local Search with 2OPT and 3OPT neighbour structure.
  - Clustering algorithms.
 
Despite we reach great solutions for TSP, we want to reach great solutions for VRP, thus, in next steps we want to implement a Local Search for VRP. 
 
We compile this program in this form, python solve.py [instance] [clustering_method*]
* Clustering methods below:
  - 1 for greedypath
  - 2 for serial
  - 3 for parallel
