import pandas as pd
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import re

import sys
import os

rnd = np.random
rnd.seed(0)

def vrptw_cplex(dataframe, dataset):
    print(dataframe.head())
    depot_data = dataframe[0:1]
    clients = dataframe[1:]
    length = len(clients)
    
    n = len(clients)
    Q = 200
    N = [i for i in range(1, n + 1)]
    V = [0] + N
    q = {i: clients["DEMAND"][i] for i in N}
    velocity = 20

    loc_x = rnd.rand(len(V)) * 200  # Randomly generated data if no specific data is being used 
    loc_y = rnd.rand(len(V)) * 100
    for i in range(len(V)):         # Replacing the dataset information with the randomly generated data
        loc_x[i] = dataframe["XCOORD."][i]
        loc_y[i] = dataframe["YCOORD."][i]

    A = [(i, j) for i in V for j in V if i != j]  # List of Arcs
    c = {(i, j): round(np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j])) for i, j in A}  # Dictionary of distances/costs

    T = max(dataframe["READY_TIME"]) ## Recoding the time windows and service times 
    service_time = dict()
    tw_starts = dict()
    tw_ends = dict()
    for item in range(1, n + 1):
        tw_s = dataframe["READY_TIME"][i]
        tw_e = dataframe["DUE_DATE"][i]
        service_time[item] = dataframe["SERVICE_TIME"][i]
        tw_starts[item] = tw_s
        tw_ends[item] = tw_e

    # Create a CPLEX model:
    mdl = Model('CVRP')

    # Define arcs and capacities:
    x = mdl.binary_var_dict(A, name='x')
    u = mdl.continuous_var_dict(N, ub=Q, name='u')
    t = mdl.continuous_var_dict(N, ub=T, name='t')
    a = mdl.integer_var_dict(tw_starts, name='a')
    e = mdl.integer_var_dict(tw_ends, name='e')
    s = mdl.integer_var_dict(service_time, name='s')

    # Define objective function:
    mdl.minimize(mdl.sumsq(c[i, j] * x[i, j] for i, j in A))

    # Add constraints:
    mdl.add_constraints(mdl.sum(x[i, j] for j in V if j != i) == 1 for i in N)  # Each point must be visited
    mdl.add_constraints(mdl.sum(x[i, j] for i in V if i != j) == 1 for j in N)  # Each point must be left
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j], u[i] + q[j] == u[j]) for i, j in A if i != 0 and j != 0)
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j], mdl.max(t[i] + c[i, j] + s[i] / 30, a[j]) == t[j]) for i, j in A if i != 0 and j != 0)
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j],  e[j]>=t[j]) for i, j in A if i != 0 and j != 0)
    mdl.add_constraints(u[i] >= q[i] for i in N)

    mdl.parameters.timelimit = 1*60*60 # Add running time limit of one hour

    # Solving model:
    solution = mdl.solve(log_output=True)

    print(solution.solve_status)  # Returns if the solution is Optimal or just Feasible

    active_arcs = [a for a in A if x[a].solution_value > 0.9]
    print("Dataset=>"+ dataset + " " + str(length))
    print(active_arcs) # Returns the arcs of the found solution
    
    # Plot solution:
    title = "Dataset=>"+ dataset + " " + str(length)
    plt.title(title)
    plt.scatter(loc_x[1:], loc_y[1:], c='b')
    for i in N:
        plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
    for i, j in active_arcs:
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)
    plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
    plt.axis('equal')
    plt.savefig("Figure_"+ dataset + "_" + str(length)+".jpg")
    plt.show()    


if __name__ == '__main__':
    dataset_name = "rc101"
    length = 15
    print(dataset_name, length)
    print(f"Dataset: {dataset_name}, Length = {length}")
    file = pd.read_csv("./Data/"+dataset_name+".txt", delim_whitespace=True)
    vrptw_cplex(file, dataset_name, length)






