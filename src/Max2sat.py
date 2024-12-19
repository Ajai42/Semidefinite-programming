import random
import itertools





def generate_max_2_sat(N, M):
    """
    Generates all unique Max-2-SAT clauses for N variables, picks M random clauses.
    
    Args:
        N (int): Number of variables.
        M (int): Number of clauses to pick randomly.
    
    Returns:
        list: List of M randomly picked Max-2-SAT clauses in CNF format.
    """
    # Generate all positive and negative literals
    literals = list(range(1, N+1)) + list(range(-1, -N-1, -1))
    
    # Generate all unique two-literal clauses using permutations
    all_clauses = list(itertools.combinations(literals, 2))
    
    # Randomly sample M clauses
    selected_clauses = random.sample(all_clauses, M)
    
    return selected_clauses


def violated_clauses(instance, solution):
    """
    Counts how many clauses are violated in a Max-2-SAT instance given a solution in ±1 format.

    Args:
        instance (list of tuples): Each tuple represents a clause with two literals, e.g., [(1, -2), (-1, 2)].
        solution (list of int): Solution in ±1 format (e.g., [1, -1, 1] for x1=True, x2=False, x3=True).

    Returns:
        int: The number of violated clauses.
    """
    violated_count = 0

    for clause in instance:
        # Evaluate the literals in the clause
        literal1, literal2 = clause

        # Get the truth values of the literals based on the ±1 solution
        value1 = solution[abs(literal1) - 1] if literal1 > 0 else -solution[abs(literal1) - 1]
        value2 = solution[abs(literal2) - 1] if literal2 > 0 else -solution[abs(literal2) - 1]

        # Clause is violated if both literals evaluate to -1 (False)
        if value1 == -1 and value2 == -1:
            violated_count += 1 
        

    return violated_count



