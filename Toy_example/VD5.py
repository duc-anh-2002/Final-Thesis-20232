import numpy as np
from scipy.optimize import minimize
from autograd import grad
import matplotlib.pyplot as plt
import torch, math, random 
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import os 
from math import e
from pytictoc import TicToc
Kappa = 0.5
k = 0
alpha = 0
eps = 0.4
delta = 0.3
num_of_component_functions = 6

def J_delta(x, ref):
    J_delta = []
    for i in range (1, num_of_component_functions + 1):
        # print(f(x, i, ref))
        if (f(x, i, ref) >= F(x, ref) - delta):
            J_delta.append(i)
    return J_delta

#3 dims, 3 objectives
def f1(x):
    return x[0]**2 + x[1]**2 + x[2]**2 -1
def f2(x):
    return x[0]**2 + x[1]**2 + (x[2]-2)**2
def f3(x):
    return x[0] + x[1] + x[2] - 1 
def f4(x):
    return x[0] + x[1] - x[2] + 1
def f5(x):
    return 2*x[0]**3 + 6*x[1]**2 + 2*(5*x[2]-x[0]+1)**2
def f6(x):
    return x[0]**2 - 9*x[2]

# calculate the gradients using autograd
f1_dx = grad(f1)
f2_dx = grad(f2)
f3_dx = grad(f3)
f4_dx = grad(f4)
f5_dx = grad(f5)
f6_dx = grad(f6)

def f(x, i, ref):
    if i == 1:
        return x[0]**2 + x[1]**2 + x[2]**2 -1
    if i == 2:
        return x[0]**2 + x[1]**2 + (x[2]-2)**2
    if i == 3:
        return x[0] + x[1] + x[2] - 1 
    if i == 4:
        return x[0] + x[1] - x[2] + 1
    if i == 5:
        return 2*x[0]**3 + 6*x[1]**2 + 2*(5*x[2]-x[0]+1)**2
    if i == 6:
        return x[0]**2 - 9*x[2]
     
def F(x, ref):
    return max(f(x, i, ref) for i in range (1, num_of_component_functions + 1))

def objective(X, x, ref):
    return X[0] + 1/2 * np.linalg.norm(X[1:])**2           

def constraint1(X):
    y1 = f(x, 1, ref[0])
    return -1*(np.dot(f1_dx(x.detach().numpy()), X[1:]) + y1.detach().numpy() - X[0])

def constraint2(X): 
    y2 = f(x, 2, ref[0])
    return -1*(np.dot(f2_dx(x.detach().numpy()), X[1:]) + y2.detach().numpy() - X[0])

def constraint3(X): 
    y3 = f(x, 3, ref[0])
    return -1*(np.dot(f3_dx(x.detach().numpy()), X[1:]) + y3.detach().numpy() - X[0])

def constraint4(X): 
    y4 = f(x, 4, ref[0])
    return -1*(np.dot(f4_dx(x.detach().numpy()), X[1:]) + y4.detach().numpy() - X[0])

def constraint5(X): 
    y5 = f(x, 5, ref[0])
    return -1*(np.dot(f5_dx(x.detach().numpy()), X[1:]) + y5.detach().numpy() - X[0])

def constraint6(X): 
    y6 = f(x, 6, ref[0])
    return -1*(np.dot(f6_dx(x.detach().numpy()), X[1:]) + y6.detach().numpy() - X[0])


def solve_p(X, x, ref): 
    k_delta = J_delta(x, ref) 
    constraint = []
    if 1 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint1})
    if 2 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint2})
    if 3 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint3})
    if 4 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint4})
    if 5 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint5})
    if 6 in k_delta:
        constraint.append({'type':'ineq', 'fun': constraint6})
    res = minimize(objective, x0= X, args = (x, ref), constraints=constraint, tol=1e-6)
    return res.x 

def gradient_1(X, x, ref):
    Kappa_new = 1
    while (F(x.detach().numpy() + Kappa_new * np.array(X[1:]), ref) > F(x.detach().numpy(), ref) - Kappa_new * eps * np.linalg.norm(X[1:])**2):
        Kappa_new = Kappa_new * Kappa
    x_new = x.detach().numpy() + Kappa_new * np.array(X[1:])
    # print('Kappa = ', Kappa_new)
    return x_new

def gradient(X, x, ref, kappa, sigma, eta, iter):
    if (F(x.detach().numpy() + kappa * np.array(X[1:]), ref) <= F(x.detach().numpy(), ref) - kappa * eps * np.linalg.norm(X[1:])**2):
        kappa =  kappa + eta ** iter
    else:
        kappa = kappa * sigma
    x_new = x.detach().numpy() + kappa * np.array(X[1:])
    return x_new, kappa

def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x), f3(x), f4(x), f5(x), f6(x)]), np.stack([f1_dx(x), f2_dx(x), f3_dx(x), f4_dx(x), f5_dx(x), f6_dx(x)])

### create the ground truth Pareto front ###
def create_pf_concave():
    ps = np.linspace(-1/np.sqrt(2),1/np.sqrt(2))
    pf = []

    for x1 in ps:
        for x2 in ps:
            for x3 in ps:
        #generate solutions on the Pareto front:
                x = np.array([x1,x2, x3])

                f, f_dx = concave_fun_eval(x)

                pf.append(f)
    pf = np.array(pf)
    return pf

def create_pf1():
    ps1 = np.linspace(-3.5, 3.5, num=1000)
    pf = []
    for x1 in ps1:
        for x2 in ps1:
            x = [x1, x2]
            f = [f1(x), f2(x)]
        #print(f)
            pf.append(f)
    pf = np.array(pf)
    return pf

if __name__ == "__main__":
    n_dim = 3
    b = 2.0
    p = np.random.uniform(-0.5,0.5,n_dim)
    X = [b, *p]
    res = []
    ref = []

    # non-monotone
    sigma = 0.9
    eta = 0.01
    ref = np.array([[1, 1, 1]])
    
    t = TicToc()
    t.tic()
    x0 = np.array([1.0, 1.0, 1.0])
    x = torch.tensor(x0, requires_grad=True)
    kappa = 1
    for iter in range (50):
        X_new = solve_p(X, x, ref[0])
        # x_new = gradient_1(X_new, x, ref[0])
        x_new, kappa_new = gradient(X_new, x, ref[0], kappa, sigma, eta, iter)
        kappa = kappa_new
        x = torch.tensor(x_new, requires_grad=True)
        X = [F(x, ref[0]).detach().numpy(), *X_new[1:]]
    res.append([f(x_new, 1, ref[0]), f(x_new, 2, ref[0]), f(x_new, 3, ref[0]), f(x_new, 4, ref[0]), f(x_new, 5, ref[0]), f(x_new, 6, ref[0])])
    t.toc()
    print('Initial point (1,1,1), algo 1 = ', x_new)
    print('Objective value, algo 1 = ', max([f(x_new, 1, ref[0]) , f(x_new, 2, ref[0]), f(x_new, 3, ref[0]), f(x_new, 4, ref[0]), f(x_new, 5, ref[0]), f(x_new, 6, ref[0])]))
    
    X = [b, *p]
    # res = []
    t.tic()
    x0 = np.array([100.0, 100.0, 100.0])
    x = torch.tensor(x0, requires_grad=True)
    kappa = 1
    for iter in range (50):
        X_new = solve_p(X, x, ref[0])
        # x_new = gradient_1(X_new, x, ref[0])
        x_new, kappa_new = gradient(X_new, x, ref[0], kappa, sigma, eta, iter)
        kappa = kappa_new
        print('norm = ', np.linalg.norm(X_new[1:]))
        print('kappa = ', kappa)
        x = torch.tensor(x_new, requires_grad=True)
        X = [F(x, ref[0]).detach().numpy(), *X_new[1:]]
    res.append([f(x_new, 1, ref[0]), f(x_new, 2, ref[0]), f(x_new, 3, ref[0]), f(x_new, 4, ref[0]), f(x_new, 5, ref[0]), f(x_new, 6, ref[0])])
    t.toc()
    print('Initial point (100,100,100), algo 1 = ', x_new)
    print('Objective value, algo 1 = ', max([f(x_new, 1, ref[0]) , f(x_new, 2, ref[0]), f(x_new, 3, ref[0]), f(x_new, 4, ref[0]), f(x_new, 5, ref[0]), f(x_new, 6, ref[0])]))
    print(res)