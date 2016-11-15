# This program illustrates a simple polynomial curve fitting example 
# Import all required libraries
import numpy as np
import math
import random
from scipy.spatial import distance
from matplotlib import pyplot as plt

# Generate N equally spaced floating point numbers
# between a and b
def generate_floats(a, b, N):
    x = np.zeros(N)
    delta = (float) (b - a)/N
    for idx in range(N):
        x[idx] = a + idx*delta
    
    return x

# Generate the system matrix for the polynomial
# function
def generate_matrix(x, y, M, N):
    # The expressions for A and b
    # follow from minimizing SE
    A = np.zeros([M, M])
    b = np.zeros([M])
    for row in range(M):
        b[row] = np.sum(np.multiply(y, np.power(x, row)))
        for col in range(M):
            A[row, col] = np.sum(np.power(x, (row+col)))  
       
    A[0,0] = N
    return A, b

# Generate samples from the fit 
def generate_y_hat(x, w, M, N):
    y_hat = np.zeros([N])
    for idx in range(N):
        y_hat[idx] = 0
        for m in range(M):
            y_hat[idx] += w[m]*np.power(x[idx],m)
    return y_hat

##################
# Set model order
M = 9 
# Set number of training samples 
N = 10
# Generate floats
x = generate_floats(0, 2*np.pi, N) 
# Generate noise
mean = 0
std = 0.05
# Generate some numbers from the sine function
y = np.sin(x)
# Add noise
y += np.random.normal(mean, std, N) 
# Display results
plt.plot(x, y, 'o')
# Find the matrix A and the vector b 
A, b = generate_matrix(x, y, M, N)
# Find the optimal solution w
w = np.dot(np.linalg.inv(A), b)
# Set number of training samples 
N_fit = 100
# Generate floats
x = generate_floats(0, 2*np.pi, N_fit) 
# Generate samples from the polynomial
y_hat = generate_y_hat(x, w, M, N_fit)
plt.plot(x, y_hat, 'g')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Polynomial model order: %s, number of training samples: %s' %(M,N))
plt.show()

