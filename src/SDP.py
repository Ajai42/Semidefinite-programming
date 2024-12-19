import cvxpy as cp
import numpy as np
from src.Max2sat import violated_clauses

def SDP_max_sat(N, instance, solver="Default"):
    """
    Solves a Max-2-SAT problem using a semidefinite programming (SDP) relaxation

    Args:
        N (int): Number of variables in the Max-2-SAT problem.
        instance (ndarray): A 2D array where each row represents a clause as two literals 
                             (e.g., [[1, -2], [-3, 4]]). The literals can be positive or negative 
                             integers representing variable indices.
        solver (str, optional): The solver to use for solving the SDP. Options are 'Default' or 'Mosek'.
                                 Default is 'Default'.

    Returns:
        ndarray: The solution matrix `Y`, representing the relaxed SDP solution.

    Methodology:
        1. Construct a semidefinite programming relaxation for the Max-2-SAT problem:
           - Introduce a positive semidefinite (PSD) variable matrix `Y`.
           - Define the objective function based on SDP relaxation terms involving the clauses.
           - Add diagonal constraints to enforce vector normalization.
        2. Solve the SDP problem using the chosen solver.
\
    """
    # Define SDP variable
    Y = cp.Variable((N + 1, N + 1), PSD=True)

    # Extract clause weights and variable indices
    weights = np.sign(instance)
    variables = np.abs(instance)

    indices_0 = variables[:, 0]  # First variable in each clause
    indices_1 = variables[:, 1]  # Second variable in each clause
    w0 = weights[:, 0].reshape(1, -1)  # Weights for the first variables
    w1 = weights[:, 1].reshape(1, -1)  # Weights for the second variables

    # Precompute indexed slices for Y
    Y_0_indices_0 = cp.vstack([Y[0, i] for i in indices_0]).T
    Y_0_indices_1 = cp.vstack([Y[0, i] for i in indices_1]).T
    Y_indices_0_1 = cp.vstack([Y[i, j] for i, j in zip(indices_0, indices_1)]).T

    # Precompute w0 * w1
    w0_w1 = np.multiply(w0, w1)

    # Define constraints for Y
    constraints = [cp.diag(Y) == 1]

    # Define the objective function for SDP relaxation
    terms = (1 / 4) * (
        3
        + cp.multiply(w0, Y_0_indices_0)
        + cp.multiply(w1, Y_0_indices_1)
        - cp.multiply(w0_w1, Y_indices_0_1)
    )
    objective = cp.sum(terms)

    # Solve the SDP problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    if solver == "default":
        prob.solve()
    elif solver == "Mosek":
        prob.solve(solver=cp.MOSEK)  # Use MOSEK solver

    return Y.value


def rounding_87(Y, N, instance, h_planes=10000):
    """
    Generates random solutions to a Max-2-SAT problem using hyperplane rounding on the SDP solution.

    Args:
        Y (ndarray): The solution matrix from the SDP relaxation of the Max-2-SAT problem.
        N (int): Number of variables in the Max-2-SAT problem.
        instance (ndarray): A 2D array where each row represents a clause as two literals 
                             (e.g., [[1, -2], [-3, 4]]). The literals can be positive or negative 
                             integers representing variable indices.
        h_planes (int, optional): Number of hyperplanes used for rounding. Default is 10000.

    Returns:
        list: A list containing the number of violated clauses (NVC) for each random solution generated.

    Methodology:
        1. Use the solution matrix `Y` obtained from SDP relaxation.
        2. Perform hyperplane rounding: 
           - Generate random vectors and project the SDP solution to produce ±1 solutions.
        3. Count the number of violated clauses for each random solution using the `violated_clauses` function.
    """
    # Initialize list for storing number of violated clauses
    NVC = []
    N_round = h_planes

    # Perform hyperplane rounding and evaluate solutions
    for _ in range(N_round):
        u = np.random.randn(N + 1)  # Normal vector for a random hyperplane
        eigenvalues, eigenvectors = np.linalg.eigh(Y)
        # Compute the square root
        Sqrt_y = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
        x = np.sign(Sqrt_y @ u)
        solution = x[0] * x[1:]     # Extract solution in ±1 format
        nvc = violated_clauses(instance, solution)  # Count violated clauses
        NVC.append(nvc)
        
    return NVC
