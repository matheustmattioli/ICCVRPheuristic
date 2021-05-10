#!/usr/bin/python
# -*- coding: utf-8 -*-

# Matheus Mattioli's undergraduate research at 
# Departamento de Computação (DC), Universidade Federal de São Carlos (UFSCar), Brazil.
# Supervised by Prof. Mario San Felice and Prof. Pedro Hokama.
# Solver for Capacitated Vehicle Routing Problem (CVRP) using heuristics and metaheuristics.
# Last Update 05/04/2021

import sys
import time
import math
import random
#from numba import jit
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

N_ITE_MS_CLUST = 100 # Number of iterations multistart cluster.
MAX_ITER_W_NO_IMPROV = 100
N_ITE_GRASP = 500 # Number of iterations GRASP.
ALPHA = 0 # Dinamic alpha for RCL calculation.
ALPHA_MAX = 0.5
DEBUG = 1

# Calculate distance between two nodes.
def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


# Calculate length of a tour.
def calc_obj(vehicle_tours, customers):
    obj = 0
    if len(vehicle_tours) > 0:
        for v in range(len(vehicle_tours)):
            obj += length(customers[vehicle_tours[v - 1]], customers[vehicle_tours[v]])
    return obj

# Making clusters of customers with greedy path technique. In
# which we select a random customer to be the center of cluster and 
# do a nearest neighbour addiction until we fulfill the capacity of the vehicle. 
def greedyPathClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, customers_visited):
    cluster = [] # Subset of customers
    aux = 1
    cluster.append(int(0))
    capacity_remaining = vehicle_capacity

    flag_time = 0
    if len(dictofcustomers.keys()) > 0: # If we still have customers to attend.
        j = random.choice(list(dictofcustomers.keys())) # Choose a random customer to be the center.
        
        capacity_remaining -= customers[j].demand
        cluster.append(int(j))
        aux += 1 # aux is an iterator who appoint to the next slot in cluster list.
        customers_visited += 1

        remove = dictofcustomers.pop(j) # For debug purposes, we pop the value in "remove".
        if DEBUG > 1:
            print(remove)
        while(capacity_remaining > 0 and customers_visited <= customer_count - 1 and (time.time() - start) < 1800):
            best_value = float('inf')
            flag_capacity = 1
            for v in range(1, customer_count + 1):
                if v in dictofcustomers and capacity_remaining - customers[v].demand >= 0:
                    minv = length(customers[cluster[aux - 1]], customers[v])
                    flag_capacity = 0
                    if minv < best_value: # Select nearest neighbour
                        best_value = minv
                        customer_candidate = v
            # If no one neighbour is considered, then capacity remaining in vehicle is greater than 0, but isn't enough to attend any customer.
            if flag_capacity == 1:
                break
            # Insert selected customer in cluster
            cluster.append(int(customer_candidate))
            aux += 1
            capacity_remaining -= customers[customer_candidate].demand 

            remove = dictofcustomers.pop(customer_candidate)
            if DEBUG > 1:
                print(remove)

            customers_visited += 1

    cluster.append(int(0))
    if time.time() - start >= 1800:
        flag_time = 1
    return cluster, customers_visited, flag_time

# Making clusters of customers choosing "vehicle_count" centers and adding
# each customer to the closest center.
def parallelClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, vehicle_count, customers_visited):
    cluster = [[] for i in range(vehicle_count)] # creating "vehicle_count" clusters.
    for i in range(vehicle_count):
        cluster[i].append(int(0))  
    capacity_remaining = [vehicle_capacity for i in range(vehicle_count)] # capacity remaining of each cluster.

    flag_time = 0
    for i in range(vehicle_count):
        if time.time() - start >= 1800:
            flag_time = 1
            break
        # if there is customers to attend.
        if len(dictofcustomers.keys()) > 0:
            j = random.choice(list(dictofcustomers.keys())) # choose a random customer to be center of cluster.
            cluster[i].append(int(j))
            customers_visited += 1
            capacity_remaining[i] -= customers[j].demand
            remove = dictofcustomers.pop(j)
            if DEBUG > 5:
                print(remove)

    # In this loop we add one customer to the closest center of a cluster each iteration.
    capacity_flag = 1
    while len(dictofcustomers.keys()) > 0 and capacity_flag >= 1 and (time.time() - start) < 1800:
        for v in customers:
            if v.index in dictofcustomers.keys():
                closest = float('inf')
                capacity_flag = 0
                for i in range(vehicle_count):
                    v_length = length(v, customers[cluster[i][1]])
                    if v_length < closest and capacity_remaining[i] - v.demand > 0:
                        closest = v_length
                        closest_vehicle = i
                        capacity_flag = 1
                if capacity_flag >= 1: # if there is capacity remaining in this especific cluster.
                    cluster[closest_vehicle].append(int(v.index))
                    customers_visited += 1
                    capacity_remaining[closest_vehicle] -= v.demand
                    remove = dictofcustomers.pop(v.index)
                    if DEBUG > 5:
                        print(remove)
    # closing each cluster with depot.
    for i in range(vehicle_count):
        cluster[i].append(int(0))         
    if time.time() - start >= 1800:
            flag_time = 1           
    return cluster, customers_visited, flag_time

# Making clusters of customers choosing the closests customers to a center. This center
# is a selected random customer. We do it until we fulfill the capacity of the vehicle.
def serialClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, customers_visited):
    cluster = [] # subset of customers
    cluster.append(int(0))
    capacity_remaining = vehicle_capacity

    flag_time = 0
    if len(dictofcustomers.keys()) > 0: # If we still have customers to attend.
        j = random.choice(list(dictofcustomers.keys())) # Choose a random customer to be the center.

        capacity_remaining -= customers[j].demand
        cluster.append(int(j))
        customers_visited += 1

        remove = dictofcustomers.pop(j) # For debug purposes, we pop the value in "remove".
        if DEBUG > 5:
            print(remove)
        
        while(capacity_remaining > 0 and customers_visited <= customer_count - 1 and (time.time() - start) < 1800):
            best_value = float('inf')
            flag = 1
            # "greedy for"
            # greedy choice: add to cluster the closest customer to center.
            for v in range(1, customer_count + 1):
                if v in dictofcustomers and capacity_remaining - customers[v].demand >= 0:
                    minv = length(customers[cluster[0]], customers[v])
                    flag = 0
                    if minv < best_value:
                        best_value = minv
                        customer_candidate = v
            # If no one neighbour is considered, then capacity remaining in vehicle is greater than 0, but isn't enough to attend any customer.
            if flag == 1: 
                break

            cluster.append(int(customer_candidate))
            capacity_remaining -= customers[customer_candidate].demand 

            remove = dictofcustomers.pop(customer_candidate)
            if DEBUG > 5:
                print(remove)
    
            customers_visited += 1
    cluster.append(int(0))
    if time.time() - start >= 1800:
        flag_time = 1
    return cluster, customers_visited, flag_time

# Constuctive heuristic for Traveling Salesman Problem (TSP).
# It's a nearest neighbour (NN) algorithm with Restricted Candidate List (RCL).
def greedypath_RCL(circuit, customers):
    # Do hamiltonian circuit with greedy choices in selected subset
    # Selected by clusters function
    # using RCL
    nodeCount = len(circuit)
    obj_BS = float('inf')
    flag_time = 0
    #for v in circuit:
    for v in range(1):
        dictofpositions = {circuit[i] : circuit[i] for i in range(nodeCount)}
        solution_greedy = [0 for i in range(nodeCount)]
        k = 0
        solution_greedy[k] = dictofpositions.pop(v)
        k += 1
        # greedy choices
        while k < nodeCount and (time.time() - start) < 1800: 
            nearest_value = float('inf')
            farthest_value = 0
            # deciding nearest and farthest customers from  k - 1 customer
            for n in circuit:
                if n in dictofpositions:
                    lengthN = length(customers[solution_greedy[k - 1]], customers[n])
                    if lengthN < nearest_value:
                        nearest_value = lengthN
                    if lengthN > farthest_value:
                        farthest_value = lengthN
            RCL = []
            # filling in RCL
            for n in circuit:
                if n in dictofpositions:
                    lengthN = length(customers[solution_greedy[k - 1]], customers[n])
                    if lengthN <= (nearest_value + (farthest_value - nearest_value)*ALPHA): # Condition to insert neighbours in RCL
                        RCL.append(n)
            solution_greedy[k] = random.choice(RCL)
            remove = dictofpositions.pop(solution_greedy[k])
            if DEBUG > 1:
                print("Vertice escolhido para a posicao k = %d", k)
                print(remove)
            k += 1
        # Decide best solution found
        curr_obj = calc_obj(solution_greedy, customers)
        if curr_obj < obj_BS:
            obj_BS = curr_obj
            best_solution = solution_greedy
        dictofpositions.clear()
    if time.time() - start >= 1800:
        flag_time = 1
    return best_solution, flag_time


# Local Search for TSP with 2OPT neighbour structure.
#@jit(nopython=True)
def localSearch2OPT(cluster, customers):
    # Small improvements in solutions through analyze of neighborhoods
    obj = calc_obj(cluster, customers) # Cost of the initial solution
    customer_count = len(cluster)
    # Initialize variables with datas from the initial solution
    lenght_BS = obj 
    lenght_solution = lenght_BS
    best_solution = list(cluster)
    count_iteration = 0 # for debug
    # Main loop
    while (time.time() - start) < 1800: 
        try:
            if DEBUG >= 2:
                start_while = time.time()
            lenght_BF = lenght_solution
            best_x = customer_count - 1
            best_y = 0
            # 2-OPT NEIGHBORHOOD
            for x in range(0, customer_count - 2):
                for y in range(x + 1, customer_count - 1):
                    edgeA = length(customers[cluster[x]], customers[cluster[x - 1]])
                    edgeB = length(customers[cluster[y]], customers[cluster[(y + 1) % customer_count]])
                    edgeC = length(customers[cluster[x]], customers[cluster[(y + 1) % customer_count]])
                    edgeD = length(customers[cluster[y]], customers[cluster[(x - 1)]])
                    lenght_PS = lenght_solution - (edgeA + edgeB) 
                    lenght_PS = lenght_PS + (edgeC + edgeD)
                    if lenght_PS < lenght_BF:
                        best_x = x
                        best_y = y
                        lenght_BF = lenght_PS
            cluster[best_x:best_y + 1] =  cluster[best_x:best_y + 1][::-1]
            lenght_solution = lenght_BF
            # Update solution
            if lenght_solution < lenght_BS:
                best_solution = list(cluster)
                lenght_BS = lenght_solution
                if DEBUG >= 2:
                    print("--------------------------------------------------------")
                    print(lenght_BS)
                    end_while = time.time()
                    count_iteration += 1
                    print("tempo do loop", end_while - start_while)
                    print("iteracao numero ", count_iteration)
            else:
                break                
        except KeyboardInterrupt:
            break
    return best_solution, lenght_BS

# Local Search for TSP with 3OPT neighbour structure.
#@jit(nopython=True)
def localSearch3OPT(cluster, customers):
    # Small improvements in solutions through analyze of neighborhoods
    nodeCount = len(cluster)
    obj = calc_obj(cluster, customers) # Cost of the initial solution
    # Initialize variables with datas from the initial solution
    if DEBUG >= 2:
        print("Valor inicial = ", obj)
    length_BS = obj 
    length_solution = length_BS
    best_solution = list(cluster)
    count_iteration = 0 
    # Main loop
    while (time.time() - start) < 1800: 
        try:
            if DEBUG >= 2:
                start_while = time.time()
            length_BF = length_solution
            combination = 0
            best_combination = 0
            best_x = nodeCount - 1
            best_z = nodeCount - 1
            best_y = 0
            # 3-OPT NEIGHBORHOOD
            for x in range(0, nodeCount - 3):
                for y in range(x + 1, nodeCount - 2):
                    for z in range(y + 1, nodeCount - 1):
                        # cut 3 edges
                        edgeA = length(customers[cluster[x]],
                                    customers[cluster[x + 1]])
                        edgeB = length(customers[cluster[y]],
                                    customers[cluster[y + 1]])
                        edgeC = length(
                            customers[cluster[z]], customers[cluster[(z + 1) % nodeCount]])
                        """ 3 arestas inseridas
                        testar as 3 combinacoes de arestas """
                        # Combination I
                        edgeD = length(customers[cluster[x]],
                                    customers[cluster[y + 1]])
                        edgeE = length(customers[cluster[z]],
                                    customers[cluster[y]])
                        edgeF = length(
                            customers[cluster[x + 1]], customers[cluster[(z + 1) % nodeCount]])
                        # Combination II
                        edgeG = length(customers[cluster[x]],
                                    customers[cluster[y]])
                        edgeH = length(
                            customers[cluster[x + 1]], customers[cluster[z]])
                        edgeI = length(
                            customers[cluster[y + 1]], customers[cluster[(z + 1) % nodeCount]])
                        # Combination III
                        edgeJ = length(customers[cluster[x]],
                                    customers[cluster[z]])
                        edgeK = length(
                            customers[cluster[y + 1]], customers[cluster[x + 1]])
                        edgeL = length(
                            customers[cluster[y]], customers[cluster[(z + 1) % nodeCount]])
                        # Combination IV
                        edgeM = length(customers[cluster[x]],
                                    customers[cluster[y + 1]])
                        edgeN = length(customers[cluster[z]],
                                    customers[cluster[x + 1]])
                        edgeO = length(customers[cluster[y]],
                                       customers[cluster[(z + 1) % nodeCount]])
                        """ Select best edges """
                        cost_decrease = edgeA + edgeB + edgeC
                        cost_increase1 = edgeD + edgeE + edgeF
                        cost_increase2 = edgeG + edgeH + edgeI
                        cost_increase3 = edgeJ + edgeK + edgeL
                        cost_increase4 = edgeM + edgeN + edgeO
                        if cost_increase1 <= cost_increase2:
                            cost_increase = cost_increase1
                            combination = 1
                        else:
                            cost_increase = cost_increase2
                            combination = 2
                        if cost_increase3 < cost_increase:
                            cost_increase = cost_increase3
                            combination = 3
                        if cost_increase4 < cost_increase:
                            cost_increase = cost_increase4
                            combination = 4
                        length_PS = length_solution - cost_decrease + cost_increase
                        if length_PS < length_BF:
                            best_x = x
                            best_y = y
                            best_z = z
                            length_BF = length_PS
                            best_combination = combination
            # Forming new circuit
            if best_combination == 1:   
                cluster = list(cluster[:best_x + 1] + cluster[best_y + 1:best_z + 1] + cluster[best_y:best_x: -1] + cluster[best_z + 1:])
            elif best_combination == 2:
                cluster = list(cluster[:best_x + 1] + cluster[best_y:best_x: -1] + cluster[best_z:best_y: -1] + cluster[best_z + 1:])
            elif best_combination == 3:
                cluster = list(cluster[:best_x + 1] + cluster[best_z:best_y: -1] + cluster[best_x + 1:best_y + 1] + cluster[best_z + 1:])
            elif best_combination == 4:
                cluster = list(cluster[:best_x + 1] + cluster[best_y + 1:best_z + 1] + cluster[best_x + 1:best_y + 1] + cluster[best_z + 1:])
            length_solution = length_BF
            # Update solution
            if length_solution < length_BS:
                best_solution = list(cluster)
                length_BS = length_solution
                if DEBUG >= 2:
                    print("--------------------------------------------------------")
                    print(length_BS)
                    end_while = time.time()
                    count_iteration += 1
                    print("tempo do loop", end_while - start_while)
                    print("iteracao numero ", count_iteration)
            else:
                break
        except KeyboardInterrupt:
            break
    if DEBUG >= 2:
        print("valor final = ", length_BS)
    return best_solution, length_BS


# Variable Neighbourhood Search (VNS)
# Change between 2-OPT and 3-OPT neighbour structure when find local minima, 
# until there is no way to improvement. 
def localSearchVNS(circuit, customers):
    best_obj = calc_obj(circuit, customers) # calc cost of initial solution
    best_circuit = list(circuit)
    while (time.time() - start) < 1800:
        try:
            circuit, obj = localSearch2OPT(best_circuit, customers)
            if obj < best_obj:
                best_obj = obj
                best_circuit = list(circuit)
            else:
                circuit, obj = localSearch3OPT(best_circuit, customers)
                if obj < best_obj:
                    best_obj = obj
                    best_circuit = list(circuit)
                else:
                    break
        except:
            break
    return best_circuit, best_obj


# Greedy Randomized Adaptative Search Procedure (GRASP) implementation with 
# Local Search 2-OPT as "Search Procedure".
def GRASP(cluster_vehicle, customers):
    best_value = float('inf')
    global ALPHA
    ALPHA = 0
    n_iter_w_no_improv = 0
    t = 0 # see the number of iterations, for debug purposes.
    if DEBUG > 1:
        print("line 344: time = ", time.time() - start)
    while ALPHA < ALPHA_MAX and n_iter_w_no_improv < MAX_ITER_W_NO_IMPROV and (time.time() - start) < 1800:
        solution_vehicle, flag_time = greedypath_RCL(cluster_vehicle[:-1], customers)
        if flag_time == 1:
            break
        #solution_vehicle, solution_obj = localSearchVNS(solution_vehicle, customers)
        solution_vehicle, solution_obj = localSearch2OPT(solution_vehicle, customers)
        n_iter_w_no_improv += 1
        if solution_obj < best_value:
            best_value = solution_obj
            best_solution_vehicle = list(solution_vehicle)
            n_iter_w_no_improv = 0
            if DEBUG > 3:
                print("improvement in iteration  ", t)
        ALPHA += ALPHA_MAX/N_ITE_GRASP
        t += 1
    return best_solution_vehicle, flag_time


# Receive input data of an instance and call the solver.
def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(
            Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    if DEBUG >= 1:
        print(f"Numero de clientes = {customer_count}")
        print(f"Numero de veiculos = {vehicle_count}")
        print(f"Capacidade dos veiculos = {vehicle_capacity}")

    if DEBUG >= 2:
        print("Lista de clientes:")
        for customer in customers:
            print(
                f"index do cliente = {customer.index}, demanda do cliente = {customer.demand}, ({customer.x}, {customer.y})")
        print()

    return VRPsolver(customer_count, vehicle_count, vehicle_capacity, customers)


# Solver for VRP.
# Divided in three steps:
#     1º: Clusterization
#     2º: Draw routes for each cluster (TSP)
#     3º: Improvements through VRP's local search. <- to do
def VRPsolver(customer_count, vehicle_count, vehicle_capacity, customers):
    best_obj = float('inf')
    flag_time = 0 # Stop the program when our time run out.
    #best_solution = []
    for k in range(N_ITE_MS_CLUST):
        if flag_time == 1:
            break
        flag_viability = 1
        solution = []
        dictofcustomers = {i : customers[i] for i in range(1, customer_count)}
        customers_visited = 0

        # Choose type of clusterization
        if int(sys.argv[2]) == 3:
            cluster_vehicle, customers_visited, flag_time = parallelClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, vehicle_count, customers_visited)
            for i in range(vehicle_count):
                if flag_time == 1:
                    break

                # GRASP
                best_solution_vehicle, flag_time = GRASP(cluster_vehicle[i], customers)
                if flag_time == 1:
                    break

                for j in range(len(best_solution_vehicle)):
                    if best_solution_vehicle[j] == 0:
                        best_solution_vehicle = best_solution_vehicle[j:] + best_solution_vehicle[:j + 1]
                        break
                solution.append(best_solution_vehicle) 
        else:
            for i in range(vehicle_count):
                # Choose type of clusterization
                if int(sys.argv[2]) == 1:
                    cluster_vehicle, customers_visited, flag_time = greedyPathClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, customers_visited)
                if int(sys.argv[2]) == 2:
                    cluster_vehicle, customers_visited, flag_time = serialClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, customers_visited)
                if flag_time == 1:
                    break
                
                # GRASP
                best_solution_vehicle, flag_time = GRASP(cluster_vehicle, customers)
                if flag_time == 1:
                    break

                for j in range(len(best_solution_vehicle)):
                    if best_solution_vehicle[j] == 0:
                        best_solution_vehicle = best_solution_vehicle[j:] + best_solution_vehicle[:j + 1]
                        break
                solution.append(best_solution_vehicle) 

        # Verify if are customers unvisited
        if customers_visited < customer_count - 1 and flag_time == 0:
            flag_viability = 0 # Flag to check viability of solution found
        
        # Calculate cost of VRP
        if flag_time == 0:
            obj = 0
            for i in range(vehicle_count):
                vehicle_tour = solution[i]
                if len(vehicle_tour) > 0:
                    for j in range(0, len(vehicle_tour) - 1):
                        obj += length(customers[vehicle_tour[j]], customers[vehicle_tour[j + 1]])
            if obj < best_obj and flag_viability == 1:
                best_obj = obj
                best_solution = solution
                if DEBUG >= 3:
                    print("Best solution in k = ", k, "loop")
        dictofcustomers.clear()

    # prepare the solution in the specified output format
    outputData = '%.2f' % best_obj + '\n'
    for v in range(0, vehicle_count):
        vehicle_tour = best_solution[v]
        outputData += " ".join([str(customers[vehicle_tour[j]].index) for j in range(0, len(vehicle_tour))]) + '\n'

    return outputData




start = time.time()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        output_data = solve_it(input_data)
        print(output_data)
        solution_file = open(file_location + ".sol", "w")
        solution_file.write(output_data)
        solution_file.close()
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')


end = time.time()
print(end - start)
