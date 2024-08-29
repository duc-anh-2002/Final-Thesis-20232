import numpy as np
from scipy.optimize import minimize
from autograd import grad
import matplotlib.pyplot as plt
import torch, math, random 
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import os 
from pytictoc import TicToc
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

Kappa = 0.5
k = 0
alpha = 0
eps = 0.4
delta = 0.3
num_of_component_functions = 2

def J_delta(x, ref):
    J_delta = []
    for i in range (1, num_of_component_functions + 1):
        # print(f(x, i, ref))
        if (f(x, i, ref) >= F(x, ref) - delta):
            J_delta.append(i)
    return J_delta

def f(x, i, ref):
    if i == 1:
        return (1/(x[0]**2 + x[1]**2 + 1))/ref[0]
    if i == 2:
        return (x[0]**2 + 3*x[1]**2 + 1)/ref[1]
     
def F(x, ref):
    return max(f(x, i, ref) for i in range (1, num_of_component_functions + 1))

def objective(X, x, ref):
    return X[0] + 1/2 * np.linalg.norm(X[1:])**2           

def constraint1(X):
    y1 = f(x, 1, ref[i])
    grad_f1 = torch.autograd.grad(y1, x, create_graph=True)
    return -1*(np.dot(grad_f1[0].detach().numpy(), X[1:]) + y1.detach().numpy() - X[0])

def constraint2(X): 
    y2 = f(x, 2, ref[i])
    grad_f2 = torch.autograd.grad(y2, x, create_graph=True)
    return -1*(np.dot(grad_f2[0].detach().numpy(), X[1:]) + y2.detach().numpy() - X[0])

def solve_p(X, x, ref): 
    k_delta = J_delta(x, ref) 
    constraint = []
    if 1 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint1})
    if 2 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint2})
    res = minimize(objective, x0= X, args = (x, ref), constraints=constraint, tol=1e-6)
    return res.x 

# Normalize the functions to range [0, 1]
def normalize_function(f, x, f_min, f_max):
    f_value = f(x)
    return (f_value - f_min) / (f_max - f_min)

def gradient_1(X, x, ref):
    Kappa_new = 1
    while (F(x.detach().numpy() + Kappa_new * np.array(X[1:]), ref) > F(x.detach().numpy(), ref) - Kappa_new * eps * np.linalg.norm(X[1:])**2):
        Kappa_new = Kappa_new * Kappa
    x_new = x.detach().numpy() + Kappa_new * np.array(X[1:])
    return x_new

# def gradient(X, x, ref, kappa, sigma, eta, iter):
#     x_new = x.detach().numpy() + kappa * np.array(X[1:])
#     if (F(x.detach().numpy() + kappa * np.array(X[1:]), ref) <= F(x.detach().numpy(), ref) - kappa * eps * np.linalg.norm(X[1:])**2):
#         kappa =  kappa + eta ** iter
#     else:
#         kappa = kappa * sigma
#     return x_new, kappa

def gradient(X, x, ref, kappa, sigma, eta, iter, count):
    x_new = x.detach().numpy() + kappa * np.array(X[1:])
    if (F(x.detach().numpy() + kappa * np.array(X[1:]), ref) <= F(x.detach().numpy(), ref) - kappa * eps * np.linalg.norm(X[1:])**2):
        kappa =  kappa + eta ** iter * sigma **count
    else:
        kappa = kappa * sigma
        count += 1
    return x_new, kappa, count

def f3(x):
    #2 dims
    return (1/(x[0]**2 + x[1]**2 + 1))
def f4(x):
    return (x[0]**2 + 3*x[1]**2 + 1)

f3_dx = grad(f3)
f4_dx = grad(f4)

def convex_fun_eval(x):
    """
    return the function values and gradient values
    """
    return np.stack([f3(x), f4(x)]), np.stack([f3_dx(x), f4_dx(x)])

### create the ground truth Pareto front ###
def create_pf_convex():
    ps1 = np.linspace(-1, 2, 200)
    # ps = np.array(np.meshgrid(ps1, ps2)).T.reshape(-1, 2)
    pf = []
    for x1 in ps1:
        for x2 in ps1: 
        #generate solutions on the Pareto front:
            x = np.array([x1,x2])
            # print('x = ', x)
            f, f_dx = convex_fun_eval(x)
            pf.append(f)
    pf = np.array(pf)
    return pf


if __name__ == "__main__":
    test_rays = get_reference_directions("das-dennis", 2, n_partitions=10).astype(
        np.float32)
    number_of_iteration = 100
    b = 2.0
    p = [-1.0, -1.0]
    X = [b, *p]
    res = []
    ref = []
    # non-monotone
    sigma = 0.9
    eta = 0.01
    for i in range (len(test_rays)):
        if (test_rays[i][0] != 0 and test_rays[i][1] != 0):
            ref.append([test_rays[i][0], test_rays[i][1]])
    ref = np.array(ref)
    t = TicToc()
    t.tic()
    for i in range (len(ref)):
        x0 = np.random.uniform(-10, 10, 2)
        x = torch.tensor(x0, requires_grad=True) 
        kappa = 1
        count = 0
        for iter in range (number_of_iteration):
            X_new = solve_p(X, x, ref[i])
            x_new, kappa_new, count_new =  gradient(X_new, x, ref[i], kappa, sigma, eta, iter, count)
            kappa = kappa_new
            count = count_new
            x = torch.tensor(x_new, requires_grad=True)
            X = [F(x.detach().numpy(), ref[i]), *X_new[1:]]
        res.append([f(x_new, 1, ref[i]) * ref[i][0], f(x_new, 2, ref[i]) * ref[i][1]])
    print(res)
    threshold = 10
    res = [p for p in res if p[0] <= threshold and p[1] <= threshold]
    print('len = ', len(res))
    t.toc()
    pf = create_pf_convex()
    plt.figure(figsize=(10, 6))
    plt.scatter(pf[:,0],pf[:,1], s=1, color = 'blue')
    plt.xlabel("$g_1(\\theta)$")
    plt.ylabel("$g_2(\\theta)$")
    plt.plot(pf[:,0],pf[:,1])
    # plt.grid(True)
    x = [p[0] for p in res]
    y = [p[1] for p in res]
    plt.plot(x, y, 'o', color = 'red')
    plt.savefig(r'/home/ubuntu/workspace/DANC/Toy_example/images/VD4_2.png')
    plt.show()

