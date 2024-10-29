import numpy as np
from scipy.optimize import minimize
from autograd import grad
import matplotlib.pyplot as plt
import torch, math, random 
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import os 

Kappa = 0.5
k = 0
alpha = 0
eps = 0.4
delta = 0.3
num_of_component_functions = 2

def J_delta(x):
    J_delta = []
    for i in range (1, num_of_component_functions + 1):
        if (f(x, i) >= F(x) - delta):
            J_delta.append(i)
    return J_delta


def J_h(x):
    J_h = []
    for i in range (1, 4):
        if (g(x, i) == 0):
            J_h.append(i)
    return J_h

def f(x, i):
    if i == 1:
        return (x[0]**2 + x[1]**2 + 3)/(1 + 2 * x[0] + 8 * x[1])
    if i == 2:
        return (x[0]**2 + x[1]**2 + 3)/(1 + 2 * x[1] + 8 * x[0])
     
def g(x, i):
    if i == 1:
        return -x[0]**2 - 2*x[0] * x[1] + 4
    if i == 2:
        return -x[0]
    if i == 3:
        return -x[1]

def F(x):
    return max(f(x, i) for i in range (1, num_of_component_functions + 1))

def objective(X, x):
    return X[0] + 1/2 * np.linalg.norm(X[1:])**2       

def constraint1(X):
    y1 = f(x, 1)
    grad_f1 = torch.autograd.grad(y1, x, create_graph=True)
    return -1*(np.dot(grad_f1[0].detach().numpy(), X[1:]) + y1.detach().numpy() - X[0])

def constraint2(X): 
    y2 = f(x, 2)
    grad_f2 = torch.autograd.grad(y2, x, create_graph=True)
    # print(-1*(np.dot(grad_f2[0].detach().numpy(), X[1:]) + y2.detach().numpy() - X[0]))
    return -1*(np.dot(grad_f2[0].detach().numpy(), X[1:]) + y2.detach().numpy() - X[0])

def constraint_g1(X):
    # print(-1 * (g(x, 1).detach().numpy()))
    return -1 * (g(x, 1).detach().numpy())

def constraint_g2(X):
    return -1 * (g(x, 2).detach().numpy())

def constraint_g3(X):
    return -1 * (g(x, 3).detach().numpy())


def project_x(x, g):
    """
    Project x into the space constrained by g(x,1) <= 0 and g(x,2) <= 0
    
    Parameters:
    x (numpy.ndarray): The point to be projected
    g (callable): The constraint function g(x, i)
    
    Returns:
    numpy.ndarray: The projected point
    """
    def objective(y):
        return np.sum((y - x)**2)
    
    def constraints(y):
        return np.array([g(y, 1), g(y, 2), g(y, 3)])
    
    cons = {'type': 'ineq', 'fun': lambda y: -constraints(y)}
    
    result = minimize(objective, x, method='SLSQP', constraints=cons, tol=1e-6)
    
    return result.x


def solve_p(X, x): 
    k_delta = J_delta(x) 
    l_delta = J_h(x)
    constraint = []
    if 1 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint1})
    if 2 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint2})
    if 1 in l_delta:
        constraint.append({'type':'eq', 'fun': constraint_g1})
    if 2 in l_delta:
        constraint.append({'type':'eq', 'fun': constraint_g2})
    if 3 in l_delta:
        constraint.append({'type':'eq', 'fun': constraint_g3})
    res = minimize(objective, x0= X, args = (x), method='SLSQP', constraints=constraint, tol=1e-6)
    return res.x 


def gradient_1(X, x, kappa, sigma, eta, iter):
    x_new = x.detach().numpy() + kappa * np.array(X[1:])
    x_new = project_x(x_new, g)
    print(x_new)
    if (F(x_new) <= F(x.detach().numpy()) - kappa * eps * np.linalg.norm(X[1:])**2):
        kappa = kappa + eta ** iter
    else:
        kappa = kappa * sigma
    return x_new, kappa

def gradient(X, x, kappa, sigma, eta, iter):
    x_new = x.detach().numpy() + kappa * np.array(X[1:])
    if (F(x.detach().numpy() + kappa * np.array(X[1:])) <= F(x.detach().numpy()) - kappa * eps * np.linalg.norm(X[1:])**2):
        kappa =  kappa + eta ** iter
    else:
        kappa = kappa * sigma
    return x_new, kappa

def f1(x):
    #2 dims
    return 1/25*x[0]**2+1/100*(x[1]-9/2)**2
def f2(x):
    return 1/25*x[1]**2+1/100*(x[0]-9/2)**2


def create_pf1():
    ps1 = np.linspace(-6, 0, num=500)
    pf = []
    for x1 in ps1:
        x = [9*x1/(2*x1-8),9/(2-8*x1)]
        f = [f1(x), f2(x)]
        pf.append(f)
    pf = np.array(pf)
    return pf

def bbox(min_f1, max_f1, min_f2, max_f2):  
    # Draw the bounding box based on the constrained min/max values
    plt.gca().add_patch(plt.Rectangle((min_f1, min_f2), max_f1 - min_f1, max_f2 - min_f2, 
                                      fill=False, edgecolor='blue', linewidth=2, linestyle="--"))
    print(f"BBox: f1 in [{min_f1}, {max_f1}], f2 in [{min_f2}, {max_f2}]")

def get_scaled_reference_directions(num_partitions, min_f1, max_f1, min_f2, max_f2):
    # Get the normalized reference directions
    test_rays = get_reference_directions("das-dennis", 2, n_partitions=num_partitions).astype(np.float32)
    
    # Scale the reference directions to fit within the bounding box
    f1_scaled = test_rays[:, 0] * (max_f1 - min_f1) + min_f1
    f2_scaled = test_rays[:, 1] * (max_f2 - min_f2) + min_f2
    
    # Combine the scaled directions into a single array
    scaled_ref_dirs = np.column_stack((f1_scaled, f2_scaled))
    
    return scaled_ref_dirs
import time

if __name__ == "__main__":
    test_rays = get_reference_directions("das-dennis", 2, n_partitions=10).astype(
    np.float32)
    number_of_iteration = 100
    b = 2.0
    p = [-1.0, -1.0]
    X = [b, *p]
    res = []  
    # non-monotone adaptive
    sigma = 0.9
    eta = 0.01
    # constraint
    # min_f1, max_f1, min_f2, max_f2 = 0.08, 0.3, 0.12, 0.2
    min_f1, max_f1, min_f2, max_f2 = 0.08, 0.3, 0.08, 0.3
    ref = []
    time_taken_list = []
    ref = get_scaled_reference_directions(10, min_f1, max_f1, min_f2, max_f2)
    for i in range (1):
        # x0 = [float(random.randrange(-10.0, 10.0)), float(random.randrange(-10.0, 10.0))]
        # x0 = [1.0, 2.0]
        x0 = [-2.0, -10.0]
        # x0 = [7.0, 4.0]
        # print(x0)
        x = torch.tensor(x0, requires_grad=True) 
        # print(x)
        kappa = 1
        start_time = time.time()
        for iter in range (number_of_iteration):
            X_new = solve_p(X, x)
            x_new, kappa = gradient_1(X_new, x, kappa, sigma=0.95, eta=0.1, iter=iter) 
            
            x = torch.tensor(x_new, requires_grad=True)
            X = [F(x.detach().numpy()), *X_new[1:]]
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_list.append(time_taken)
        print(f"Time taken for sample {i}: {end_time - start_time} seconds")
        res.append([f(x_new, 1), f(x_new, 2)])
    print(x_new)
    print(res)
    # print(time_taken_list)
