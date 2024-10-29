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
        # print(f(x, i, ref))
        if (f(x, i) >= F(x) - delta):
            J_delta.append(i)
    return J_delta

def f(x, i):
    if i == 1:
        return (1/25 * x[0]**2 + 1/100 * (x[1]-9/2)**2)
    if i == 2:
        return (1/25 * x[1]**2 + 1/100 * (x[0]-9/2)**2)
     
def g(x, i):
    if i == 1:
        return f(x, i) - 0.1
    if i == 2:
        return f(x, i) - 0.2

def h(x, i):
    if i == 1:
        return 0.08 - f(x, i)
    if i == 2:
        return 0.12 - f(x, i)

def J_h(x):
    J_h = []
    for i in range (1, num_of_component_functions + 1):
        if (g(x, i) == 0):
            J_h.append(i)
    return J_h

def F(x):
    return max(f(x, i) for i in range (1, num_of_component_functions + 1))

def objective(X, x):
    return X[0] + 1/2 * np.linalg.norm(X[1:-1])**2 + X[-1]           

def constraint1(X):
    y1 = f(x, 1)
    grad_f1 = torch.autograd.grad(y1, x, create_graph=True)
    return -1*(np.dot(grad_f1[0].detach().numpy(), X[1:-1]) + y1.detach().numpy() - X[0])

def constraint2(X): 
    y2 = f(x, 2)
    grad_f2 = torch.autograd.grad(y2, x, create_graph=True)
    return -1*(np.dot(grad_f2[0].detach().numpy(), X[1:-1]) + y2.detach().numpy() - X[0])

def constraint_g1(X):
    y1 = g(x, 1)
    grad_g1 = torch.autograd.grad(y1, x, create_graph=True)
    return -1*(np.dot(grad_g1[0].detach().numpy(), X[1:-1]) + y1.detach().numpy() - X[-1])

def constraint_g2(X): 
    y2 = g(x, 2)
    grad_g2 = torch.autograd.grad(y2, x, create_graph=True)
    return -1*(np.dot(grad_g2[0].detach().numpy(), X[1:-1]) + y2.detach().numpy() - X[-1])

def constraint_h1(X):
    y1 = h(x, 1)
    grad_h1 = torch.autograd.grad(y1, x, create_graph=True)
    return -1*(np.dot(grad_h1[0].detach().numpy(), X[1:-1]) + y1.detach().numpy() - X[-1])

def constraint_h2(X):
    y2 = h(x, 2)
    grad_h2 = torch.autograd.grad(y2, x, create_graph=True)
    return -1*(np.dot(grad_h2[0].detach().numpy(), X[1:-1]) + y2.detach().numpy() - X[-1])

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
    # constraint.append({'type':'ineq', 'fun': constraint_g1})
    # constraint.append({'type':'ineq', 'fun': constraint_g2})
    res = minimize(objective, x0= X, args = (x), method='SLSQP', constraints=constraint, tol=1e-6)
    return res.x 


def project_x(x, g):
    def objective(y):
        return np.sum((y - x)**2)
    def constraints(y):
        return np.array([g(y, 1), g(y, 2)])
    cons = {'type': 'ineq', 'fun': lambda y: -constraints(y)}
    result = minimize(objective, x0= x, method='SLSQP', constraints=cons)
    return result.x

def constraint_g1(X):
    # print(-1 * (g(x, 1).detach().numpy()))
    return -1 * (g(x, 1).detach().numpy())

def constraint_g2(X):
    return -1 * (g(x, 2).detach().numpy())

def gradient_1(X, x):
    Kappa_new = 1
    while (F(x.detach().numpy() + Kappa_new * np.array(X[1:-1])) > F(x.detach().numpy()) - Kappa_new * eps * np.linalg.norm(X[1:-1])**2):
        Kappa_new = Kappa_new * Kappa
    x_new = x.detach().numpy() + Kappa_new * np.array(X[1:-1])
    # x_new = project_x(x_new, g)
    return x_new

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
    number_of_iteration = 3
    b = 2.0
    p = [-1.0, -1.0]
    gamma = 0.5
    X = [b, *p, gamma]
    res = []  
    # non-monotone adaptive
    sigma = 0.9
    eta = 0.01
    # constraint
    min_f1, max_f1, min_f2, max_f2 = 0.08, 0.3, 0.12, 0.2
    # min_f1, max_f1, min_f2, max_f2 = 0.08, 0.3, 0.08, 0.3
    ref = []
    ref = get_scaled_reference_directions(10, min_f1, max_f1, min_f2, max_f2)
    for i in range (10):
        x0 = [float(random.randrange(-10.0, 10.0)), float(random.randrange(-10.0, 10.0))]
        x = torch.tensor(x0, requires_grad=True) 
        kappa = 1
        start_time = time.time()
        for iter in range (number_of_iteration):
            X_new = solve_p(X, x)
            x_new = gradient_1(X_new, x) 
            x = torch.tensor(x_new, requires_grad=True)
            X = [F(x.detach().numpy()), *X_new[1:]]
        res.append([f(x_new, 1), f(x_new, 2)])
        end_time = time.time()
        print(f"Time taken for sample {i}: {end_time - start_time} seconds")
    print(res)
    # pf = create_pf1()
    # plt.xlabel("f1(x)")
    # plt.ylabel("f2(x)")
    # plt.plot(pf[:,0],pf[:,1])
    # x = [p[0] for p in res]
    # y = [p[1] for p in res]
    # plt.plot(x, y, 'o', color = 'red')
    # bbox(min_f1, max_f1, min_f2, max_f2)
    # plt.savefig(r'/home/ubuntu/workspace/DANC/Toy_example/constraint/images/VD1_2.png')
    # plt.show()

