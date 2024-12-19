import numpy as np
from pysat.examples.rc2 import RC2
from pysat import formula
from src.Max2sat import violated_clauses

def rc2_solver(N, instance):
    """
    Solves a Max-2-SAT problem using the RC2 algorithm 
    from the PySAT library and calculates the number of violated clauses (NVC) 
    for the solution.

    Args:
        N (int): Number of variables in the Max-2-SAT problem.
        instance (ndarray): A 2D array where each row represents a clause as two literals 
                             (e.g., [[1, -2], [-3, 4]]). The literals can be positive or negative 
                             integers representing variable indices.
    
    Returns:
        tuple: A tuple containing:
            - solution (ndarray): A 1D array representing the solution, where each entry is either 
                                   1 or -1 based on the values assigned to the variables.
            - solver_nvc (int): The number of violated clauses for the generated solution, computed 
                                 by the `violated_clauses` function.

    Methodology:
        1. Convert the Max-2-SAT problem instance into a weighted CNF (WCNF) formula using PySAT.
        2. Apply the RC2 algorithm to solve the WCNF formula.
        3. If the RC2 algorithm returns fewer variables than expected (N), append additional ones to complete the solution.
        4. Convert the solution into a ±1 format (sign representation) and evaluate the number of violated clauses.
    """
    # Convert the Max-2-SAT instance into a WCNF formula
    exprn = instance
    c = formula.WCNF()
    c.extend(exprn, weights=tuple(1 for _ in range(len(instance))))

    # Solve the WCNF formula using the RC2 algorithm
    prob = RC2(c)
    model = prob.compute()  # Get the model (solution)
    x = len(model)

    # If the model has fewer variables than N, append additional ones
    if x < N:
        for j in range(x, N):
            model.append(1)

    # Convert the model to a ±1 solution
    solution = np.sign(model)

    # Compute the number of violated clauses using the `violated_clauses` function
    solver_nvc = violated_clauses(instance, solution)
    
    return solution, solver_nvc
