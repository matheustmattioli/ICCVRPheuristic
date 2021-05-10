import sys
import fileinput
from os import listdir
from os.path import isfile, join
import math
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


DEBUG = 2


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def parse_input(instance_path):
    instance_file = open(instance_path, "r")
    input_data = instance_file.read()
    instance_file.close()

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

    instance_sol_file = open(instance_path + ".sol", "r")
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

    return check_feasibility(customer_count, vehicle_count, vehicle_capacity, customers, sol_value, routes)


def check_feasibility(customer_count, vehicle_count, vehicle_capacity, customers, sol_value, routes):

    if (vehicle_count != len(routes)):
        print("Solucao nao eh valida, pois numero de rotas na solucao esta incorreto")
        exit(0)

    num_customers_served = sum([len(route)
                                for route in routes]) - 2 * vehicle_count

    if (customer_count != num_customers_served + 1):
        print("Solucao nao eh valida, pois numero de clientes na solucao esta incorreto")
        exit(0)

    list_aux = []
    for route in routes:
        if (route[0] != 0 or route[-1] != 0):
            print("Solucao nao eh valida, pois rota nao comeca ou termina no deposito")
            exit(0)
        list_aux += route[1:-1]

    for route in routes:
        route_demand = 0
        for i in route:
            route_demand += customers[i].demand
        if (route_demand > vehicle_capacity):
            print("Solucao nao eh valida, pois rota estoura capacidade do veiculo")
            exit(0)

    list_aux.sort()
    for i in range(0, num_customers_served - 1):
        if list_aux[i] == list_aux[i+1]:
            print("Solucao nao eh valida, pois existe cliente repetido nas rotas")
            exit(0)

    obj = 0
    for route in routes:
        for i in range(len(route) - 1):
            obj += length(customers[route[i]], customers[route[i+1]])

    if sol_value != round(obj, 2):
        print(
            "Erro! Custo das rotas nao corresponde ao indicado na solucao.")
        exit(0)

    # it is a feasible solution =D
    return sol_value


if __name__ == '__main__':
    if len(sys.argv) > 1:
        instance_path = sys.argv[1].strip()
        print(instance_path)
    else:
        print('This verifier requires an input file and a test type (0-2).  Please select one from the data directory (i.e. python verifier.py ./data/vrp_16_3_1 2)')
        exit(0)
    # instance_path = input().rstrip()

    sol_value = parse_input(instance_path)

    test_type = int(sys.argv[2])
    # test_type = int(input())

    if test_type == 0:
        print("Solucao eh valida.")
        exit(0)

    instance_list = ['vrp_16_3_1', 'vrp_26_8_1', 'vrp_51_5_1',
                     'vrp_101_10_1', 'vrp_200_16_1', 'vrp_421_41_1']
    good_values = [387, 1019, 713, 1193, 3719, 2392]
    great_values = [279, 622, 525, 829, 1400, 2000]

    instance_path_list = instance_path.split('\\')
    instance_name = instance_path_list[-1]
    i = instance_list.index(instance_name)

    if test_type == 1:
        if (sol_value <= good_values[i]):
            print("Parabens! Solucao eh boa.")
        else:
            print(
                f"Solucao nao eh boa, pois nao atingiu valor {good_values[i]}")

    if test_type == 2:
        if (sol_value <= great_values[i]):
            print("Parabens! Solucao parece ser otima.")
        else:
            print(
                f"Solucao nao eh otima, pois nao atingiu valor {great_values[i]}")
