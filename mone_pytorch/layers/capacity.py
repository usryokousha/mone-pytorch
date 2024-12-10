import numpy as np
from scipy.optimize import minimize

def compute_capacity_distribution(E, e_c, delta=2.0, beta=10.0):
    """
    Solve the capacity distribution optimization problem:
    
    maximize    sum_{i=1}^E c_i * delta^(i-1) - beta * sum_{i=1}^E c_i * log(c_i)
    subject to: sum_{i=1}^E c_i = 1
                sum_{i=1}^E c_i / 2^(E-i) = e_c
                0 <= c_i <= 1
    
    Parameters
    ----------
    E : int
        Number of experts.
    e_c : float
        Effective capacity (0 < e_c < 1).
    delta : float, optional
        Weighting factor that incentivizes usage of larger experts. Default is 2.0.
    beta : float, optional
        Entropy coefficient that promotes uniformity. Default is 10.0.
    
    Returns
    -------
    c : ndarray
        Optimal capacity distribution array of shape (E,).
    """
    
    # Initial guess: uniform distribution
    c0 = np.ones(E) / E

    def objective(c):
        # sum_{i=1}^E c_i * delta^(i-1)
        term1 = sum(c[i] * (delta**i) for i in range(E))
        
        # Avoid log(0) by adding a tiny epsilon
        eps = 1e-15
        term2 = sum(c[i] * np.log(c[i] + eps) for i in range(E))
        
        # Objective to maximize: term1 - beta * term2
        # We'll minimize the negative: -(term1 - beta * term2) = -term1 + beta * term2
        return -(term1 - beta * term2)
    
    # Constraints:
    # sum(c_i) = 1
    def constraint_sum(c):
        return np.sum(c) - 1.0

    # sum(c_i / 2^(E-i)) = e_c
    # Note: For i in Python 0-based: c[i] = c_(i+1), so exponent is (E-(i+1)) = E-i-1
    def constraint_ec(c):
        return np.sum(c[i] / (2.0**(E - i - 1)) for i in range(E)) - e_c

    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'eq', 'fun': constraint_ec}
    ]

    # Bounds for each c_i: [0, 1]
    bounds = [(0.0, 1.0) for _ in range(E)]
    
    # Solve the optimization problem
    result = minimize(objective, c0, method='SLSQP', constraints=constraints, bounds=bounds)
    
    if not result.success:
        raise ValueError("Optimization did not converge: " + result.message)

    return result.x

if __name__ == "__main__":
    E = 4
    e_c = 0.9
    delta = 2.0
    beta = 10.0

    c = compute_capacity_distribution(E, e_c, delta, beta)
    print("Optimal capacity distribution c:", c)
    print("Sum of capacities:", np.sum(c))
    # Check the effective capacity constraint
    check_ec = np.sum([c[i] / 2.0**(E - i - 1) for i in range(E)])
    print("Computed e_c:", check_ec)
