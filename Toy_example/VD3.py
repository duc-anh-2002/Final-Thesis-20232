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
eps = 0.9
delta = 0.3
num_of_component_functions = 2

def J_delta(x, ref):
    J_delta = []
    for i in range (1, num_of_component_functions + 1):
        # print(f(x, i, ref))
        if (f(x, i, ref) >= F(x, ref) - delta):
            J_delta.append(i)
    return J_delta

def f1(x):

    n = len(x)

    sum1 = np.sum([(x[i] - 1.0/np.sqrt(n)) ** 2 for i in range(n)])

    f1 = 1 - e**(- sum1)
    return f1

def f2(x):

    n = len(x)

    sum2 = np.sum([(x[i] + 1.0/np.sqrt(n)) ** 2 for i in range(n)])

    f2 = 1 - e**(- sum2)

    return f2
# calculate the gradients using autograd
f1_dx = grad(f1)
f2_dx = grad(f2)

def f(x, i, ref):
    n = len(x)
    # print(x[1].item())
    if i == 1:
        sum1 = np.sum([(x[i].item() - 1.0/np.sqrt(n)) ** 2 for i in range(n)])
        return torch.tensor((1 - e**(- sum1))/ref[0], requires_grad=True)
    if i == 2:
        sum2 = np.sum([(x[i].item() + 1.0/np.sqrt(n)) ** 2 for i in range(n)])
        return torch.tensor((1 - e**(- sum2))/ref[1], requires_grad=True)
     
def F(x, ref):
    return max(f(x, i, ref) for i in range (1, num_of_component_functions + 1))

def objective(X, x, ref):
    return X[0] + 1/2 * np.linalg.norm(X[1:])**2           

def constraint1(X):
    y1 = f(x, 1, ref[i])
    return -1*(np.dot(f1_dx(x.detach().numpy()), X[1:]) + y1.detach().numpy() - X[0])

def constraint2(X): 
    y2 = f(x, 2, ref[i])
    return -1*(np.dot(f2_dx(x.detach().numpy()), X[1:]) + y2.detach().numpy() - X[0])

def solve_p(X, x, ref): 
    k_delta = J_delta(x, ref) 
    constraint = []
    # if 1 in k_delta:
    constraint.append({'type':'ineq', 'fun': constraint1})
    # if 2 in k_delta:
    constraint.append({'type':'ineq', 'fun': constraint2})
    res = minimize(objective, x0= X, args = (x, ref), constraints=constraint, tol=1e-6)
    return res.x 

def gradient_1(X, x, ref, kappa, sigma, eta, iter):
    x_new = x.detach().numpy() + kappa * np.array(X[1:])
    if (F(x.detach().numpy() + kappa * np.array(X[1:]), ref) <= F(x.detach().numpy(), ref) - kappa * eps * np.linalg.norm(X[1:])**2):
        kappa =  kappa + eta ** iter
    else:
        kappa = kappa * sigma
    return x_new, kappa

def gradient(X, x, ref, kappa, sigma, eta, iter, count):
    x_new = x.detach().numpy() + kappa * np.array(X[1:])
    if (F(x.detach().numpy() + kappa * np.array(X[1:]), ref) <= F(x.detach().numpy(), ref) - kappa * eps * np.linalg.norm(X[1:])**2):
        kappa =  kappa + eta ** iter * sigma **count
    else:
        kappa = kappa * sigma
        count += 1
    return x_new, kappa, count

def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])

### create the ground truth Pareto front ###
def create_pf_concave():
    ps = np.linspace(-1/np.sqrt(2),1/np.sqrt(2))
    pf = []

    for x1 in ps:
        for x2 in ps:
        #generate solutions on the Pareto front:
            x = np.array([x1,x2])

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
    n_dim = 20
    test_rays = get_reference_directions("das-dennis", 2, n_partitions=10).astype(
        np.float32)
    number_of_iteration = 100
    b = 2.0
    p = np.random.uniform(-0.5,0.5,n_dim)
    X = [b, *p]
    res = []
    ref = []
    sigma = 0.1
    for i in range (len(test_rays)):
        if (test_rays[i][0] != 0 and test_rays[i][1] != 0):
            ref.append([test_rays[i][0], test_rays[i][1]])
    ref = np.array(ref)
    t = TicToc()
    t.tic()
    
    # non-monotone
    sigma = 0.9
    eta = 0.01

    for i in range (len(ref)):
        x0 = np.random.uniform(-0.5,0.5,n_dim)
        x = torch.tensor(x0, requires_grad=True)
        kappa = 1
        count = 0
        for iter in range (number_of_iteration):
            X_new = solve_p(X, x, ref[i])
            # print("X_new_norm = ", np.linalg.norm(X_new[1:]))
            # X_new[1:] = X_new[1:]/np.linalg.norm(X_new[1:])
            x_new, kappa_new, count_new =  gradient(X_new, x, ref[i], kappa, sigma, eta, iter, count)
            kappa = kappa_new
            count = count_new
            # print('kappa_new = ', kappa_new)
            x = torch.tensor(x_new, requires_grad=True)
            X = [F(x, ref[i]).detach().numpy(), *X_new[1:]]
        res.append([f(x_new, 1, ref[i]) * ref[i][0], f(x_new, 2, ref[i]) * ref[i][1]])
    t.toc()
    print(res)
    pf = create_pf_concave()
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    plt.plot(pf[:,0],pf[:,1])
    x = [p[0].detach().numpy() for p in res]
    y = [p[1].detach().numpy() for p in res]
    plt.plot(x, y, 'o', color = 'red')
    plt.savefig(r'/home/ubuntu/workspace/DANC/Toy_example/images/VD3_1.png')
    plt.show()