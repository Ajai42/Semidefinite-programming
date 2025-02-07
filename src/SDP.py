import cvxpy as cp
import numpy as np
from src.Max2sat import violated_clauses

def SDP_max_sat(N, instance, solver="default"):
    """
    Solves a Max-2-SAT problem using a semidefinite programming (SDP) relaxation.

    Parameters:
        N (int): Number of variables in the Max-2-SAT problem.
        instance (array-like): A 2D array where each row represents a clause as two literals.
                              Example: [[1, -2], [-3, 4]]. Positive/negative integers denote literals.
        solver (str, optional): The solver used for SDP ("default" or "Mosek"). Default is "default".

    Returns:
        numpy.ndarray: The relaxed SDP solution matrix Y of shape (N+1, N+1).
    """
    # Convert instance to NumPy array
    instance = np.asarray(instance)

    # Compute signs and absolute indices of variables
    weights = np.sign(instance)
    variables = np.abs(instance).astype(int)

    # Extract first and second literals
    indices_0 = variables[:, 0]
    indices_1 = variables[:, 1]

    # Extract weights and reshape them for broadcasting
    w0 = weights[:, 0].reshape(1, -1)
    w1 = weights[:, 1].reshape(1, -1)
    w0w1 = (w0 * w1).reshape(1, -1)

    # Convert weights to CVXPY constants
    w0_const, w1_const, w0w1_const = cp.Constant(w0), cp.Constant(w1), cp.Constant(w0w1)

    # Define the SDP variable (N+1) x (N+1) positive semidefinite matrix
    Y = cp.Variable((N + 1, N + 1), PSD=True)

    # Efficient extraction using matrix indexing
    try:
        Y0_i = cp.reshape(Y[0, indices_0], (1, -1))
        Y0_j = cp.reshape(Y[0, indices_1], (1, -1))
        Yij = cp.hstack([Y[i, j] for i, j in zip(indices_0, indices_1)])
        Yij = cp.reshape(Yij, (1, -1))  # Reshape to match dimensions
    except Exception:
        # Fallback method for extracting values
        Y0_i = cp.hstack([Y[0, i] for i in indices_0])
        Y0_j = cp.hstack([Y[0, j] for j in indices_1])
        Yij = cp.hstack([Y[i, j] for i, j in zip(indices_0, indices_1)])
        Yij = cp.reshape(Yij, (1, -1))

    # Constraint: All diagonal elements should be 1 (unit norm)
    constraints = [cp.diag(Y) == 1]

    # Objective function: Maximize the sum of clause satisfaction terms
    terms = (1/4) * (3 + cp.multiply(w0_const, Y0_i)
                        + cp.multiply(w1_const, Y0_j)
                        - cp.multiply(w0w1_const, Yij))
    objective = cp.sum(terms)

    # Solve the SDP
    prob = cp.Problem(cp.Maximize(objective), constraints)

    if solver.lower() == "mosek":
        prob.solve(solver=cp.MOSEK)
    else:
        prob.solve()

    return Y.value


def decompose_sdp(Y):
    """
    Decomposes the SDP solution matrix Y into its square root form.

    Parameters:
        Y (numpy.ndarray): The SDP matrix.

    Returns:
        numpy.ndarray: The square root of the SDP matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Y)
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T


def rounding_87(Y, N, instance, h_planes=1000):
    """
    Perform 0.87 Random Hyperplane Rounding on the SDP solution matrix Y.

    Parameters:
        Y (numpy.ndarray): The SDP solution matrix.
        N (int): Number of variables.
        instance (list): The MAX-2SAT instance.
        h_planes (int): Number of random hyperplanes used for rounding.

    Returns:
        list: Number of violated clauses for each random solution.
    """
    Sqrt_y = decompose_sdp(Y)
    NVC = []

    for _ in range(h_planes):
        u = np.random.randn(N + 1)
        x = np.sign(Sqrt_y @ u)
        solution = x[0] * x[1:]
        NVC.append(violated_clauses(instance, solution))

    return NVC


def rotation_function(theta):
    """
    Applies a basic rotation function.

    Parameters:
        theta (float): Angle in radians.

    Returns:
        float: Rotated angle.
    """
    return 0.58831458 * theta + 0.64667394


def rotate_vector(v0, vi):
    """
    Rotates vector vi relative to v0.

    Parameters:
        v0 (numpy.ndarray): Reference vector.
        vi (numpy.ndarray): Target vector.

    Returns:
        numpy.ndarray: Rotated vector.
    """
    norm_v0, norm_vi = np.linalg.norm(v0), np.linalg.norm(vi)
    if norm_v0 == 0 or norm_vi == 0:
        return v0

    cos_angle = np.dot(v0, vi) / (norm_v0 * norm_vi)
    theta_i = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    theta_i_prime = rotation_function(theta_i)

    v0_u = v0 / norm_v0
    vi_proj = vi - np.dot(vi, v0_u) * v0_u
    norm_proj = np.linalg.norm(vi_proj)

    return v0 if norm_proj == 0 else np.cos(theta_i_prime) * v0_u + np.sin(theta_i_prime) * (vi_proj / norm_proj)


def gw_rounding(Y, N, instance, h_planes=1000):
    """
    Performs Goemans-Williamson (G&W) Random Hyperplane Rounding.

    Parameters:
        Y (numpy.ndarray): The SDP solution matrix.
        N (int): Number of variables.
        instance (list): The MAX-2SAT instance.
        h_planes (int): Number of hyperplanes.

    Returns:
        list: Number of violated clauses for each random solution.
    """
    NVC = []
    S = decompose_sdp(Y)
    v0 = S[:, 0]

    for _ in range(h_planes):
        r = np.random.randn(S.shape[0])
        r /= np.linalg.norm(r)

        solution = np.sign([np.dot(v0, r) == np.dot(S[:, i], r) for i in range(1, N + 1)])
        NVC.append(violated_clauses(instance, solution))

    return NVC


def generate_skewed_hyperplane(v0, r0=2):
    """
    Generates a skewed hyperplane.

    Parameters:
        v0 (numpy.ndarray): Reference vector.
        r0 (float): Bias parameter.

    Returns:
        numpy.ndarray: Skewed hyperplane vector.
    """
    r_perp = np.random.randn(len(v0))
    r_perp -= (np.dot(r_perp, v0) / np.dot(v0, v0)) * v0
    r_perp /= np.linalg.norm(r_perp)
    return r0 * v0 + r_perp


def sample_r0():
    """
    Samples a value for r0 from a predefined distribution.

    Returns:
        float: Sampled r0 value.
    """
    probabilities = [0.0430, 0.0209, 0.0747, 0.3448, 0.5166]
    values = [None, 0.145, 0.345, 0.755, 1.635]
    choice = np.random.choice(values, p=probabilities)
    return np.random.normal() if choice is None else choice


def rot_rounding(Y, N, instance, h_planes=1000):
    """
    Perform Rotation-Based Rounding on the SDP solution matrix Y.

    Parameters:
        Y (numpy.ndarray): The SDP solution matrix.
        N (int): The number of variables.
        instance (list): The MAX-2SAT instance.
        h_planes (int): The number of random hyperplanes used for rounding.

    Returns:
        list: A list containing the number of violated clauses for each random solution.
    """
    NVC = []
    S = decompose_sdp(Y)  # Compute the square root of SDP matrix
    v0 = S[:, 0]  # Reference vector

    for _ in range(h_planes):
        r = np.random.randn(S.shape[0])
        r /= np.linalg.norm(r)  # Normalize the hyperplane vector

        solution = []
        for i in range(1, N + 1):
            vi_rot = rotate_vector(v0, S[:, i])  # Rotate the vector
            solution.append(
                1 if np.sign(np.dot(v0, r)) == np.sign(np.dot(vi_rot, r)) else -1
            )

        nvc = violated_clauses(instance, np.array(solution))
        NVC.append(nvc)

    return NVC



def skew_rounding(Y, N, instance, h_planes=1000):
    """
    Perform Skewed Hyperplane Rounding on the SDP solution matrix Y.

    Parameters:
        Y (numpy.ndarray): The SDP solution matrix.
        N (int): The number of variables.
        instance (list): The MAX-2SAT instance.
        h_planes (int): The number of random hyperplanes used for rounding.

    Returns:
        list: A list containing the number of violated clauses for each random solution.
    """
    NVC = []
    S = decompose_sdp(Y)
    v0 = S[:, 0]
    r0 = sample_r0()  # Sampled bias from the predefined distribution

    for _ in range(h_planes):
        r = generate_skewed_hyperplane(v0, r0)

        solution = np.array([
            1 if np.sign(np.dot(v0, r)) == np.sign(np.dot(S[:, i], r)) else -1
            for i in range(1, N + 1)
        ])
        nvc = violated_clauses(instance, solution)
        NVC.append(nvc)

    return NVC


def rot_skew_rounding(Y, N, instance, h_planes=1000):
    """
    Perform Combined Rotation and Skewed Hyperplane Rounding on the SDP solution matrix Y.

    Parameters:
        Y (numpy.ndarray): The SDP solution matrix.
        N (int): The number of variables.
        instance (list): The MAX-2SAT instance.
        h_planes (int): The number of random hyperplanes used for rounding.

    Returns:
        list: A list containing the number of violated clauses for each random solution.
    """
    NVC = []
    S = decompose_sdp(Y)
    v0 = S[:, 0]
    r0 = 2  # Fixed bias as per the approach

    for _ in range(h_planes):
        r = generate_skewed_hyperplane(v0, r0)

        solution = []
        for i in range(1, N + 1):
            vi_rot = rotate_vector(v0, S[:, i])
            solution.append(
                1 if np.sign(np.dot(v0, r)) == np.sign(np.dot(vi_rot, r)) else -1
            )

        nvc = violated_clauses(instance, np.array(solution))
        NVC.append(nvc)

    return NVC
