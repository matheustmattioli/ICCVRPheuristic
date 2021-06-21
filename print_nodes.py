#!/usr/bin/python
# -*- coding: utf-8 -*-

# Terminar de adaptar para o VRP
# Preciso de um algoritmo funcional para terminar de implementar
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


DEBUG = 2

def DrawGraph(G, color, p):
    nx.draw(G, pos=p, edge_color=color, node_size=10, width=2, node_color='black')  
    return 1


def solve_it(input_data, file_location):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(
        Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    dictofpositions = {i : (customers[i].x, customers[i].y) for i in range(0, customer_count)}

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

    return VRPgreedy(customer_count, vehicle_count, vehicle_capacity, customers, dictofpositions, file_location)


def VRPgreedy(customer_count, vehicle_count, vehicle_capacity, customers, dictofpositions, file_location):

    # Modify this code to run your optimization algorithm
    # Solution
    # instance_path = input("Entre com o nome da instancia: ")
    instance_sol_file = open(file_location + ".sol", "r")
    
    solution_data = instance_sol_file.read()
    instance_sol_file.close()

    # parse the solution
    lines = solution_data.split('\n')

    firstLine = lines[0].split()
    sol_value = float(firstLine[0])

    # routes = list of lists indicating the order in which the customers are visited on each route
    routes = []
    for i in range(vehicle_count):
        line = lines[i+1].split()
        routes.append(list(map(int, line)))
   
    # Output Graph
    # Draw Solution Graph
    graphSolution = nx.Graph()
    for i in range(0, customer_count):
        graphSolution.add_node(i)
    for route in routes:
        for i in range(0, len(route) - 1):
            graphSolution.add_edge(route[i], route[i + 1])
    plt.figure(1)
    DrawGraph(graphSolution, 'black', dictofpositions)
    
    if DEBUG >= 3:
        print("Number of edges: ", graphSolution.number_of_edges())
        print("Number of nodes: ", graphSolution.number_of_nodes())
    # prepare the solution in the specified output format
    output_data = sol_value
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        output_data = solve_it(input_data, file_location)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')
    plt.show()