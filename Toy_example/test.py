import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# Define the functions
def f3(x):
    # 2 dimensions
    return (1/(x[0]**2 + x[1]**2 + 1))

def f4(x):
    return (x[0]**2 + 3*x[1]**2 + 1)

# Normalize the functions to range [0, 1]
def normalize_function(f, x, f_min, f_max):
    f_value = f(x)
    return (f_value - f_min) / (f_max - f_min)

# Compute gradients using autograd
f3_dx = grad(f3)
f4_dx = grad(f4)

# Evaluate the functions and their gradients
def convex_fun_eval(x):
    f3_min, f3_max = 0, 1  # f3 is already in the range [0, 1]
    f4_min = f4(np.array([-1, -1]))
    f4_max = f4(np.array([2, 2]))
    f3_normalized = normalize_function(f3, x, f3_min, f3_max)
    f4_normalized = normalize_function(f4, x, f4_min, f4_max)
    return np.stack([f3_normalized, f4_normalized]), np.stack([f3_dx(x), f4_dx(x)])

# Create the Pareto front
def create_pf_convex():
    ps1 = np.linspace(-1, 2, 100)
    pf = []
    for x1 in ps1:
        for x2 in ps1:
            x = np.array([x1, x2])
            f, f_dx = convex_fun_eval(x)
            pf.append(f)
    pf = np.array(pf)
    return pf

# Generate Pareto front data
pf = create_pf_convex()

# Print some sample values for debugging
print("Sample values from the Pareto front:")
print(pf[:10])

# Plot the Pareto front
plt.figure(figsize=(10, 6))
plt.scatter(pf[:, 0], pf[:, 1], s=1, c='blue')
plt.xlabel('f3(x)')
plt.ylabel('f4(x)')
plt.title('Pareto Front for normalized f3 and f4')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig(r'/home/ubuntu/workspace/DANC/Toy_example/images/test.png')
plt.show()
