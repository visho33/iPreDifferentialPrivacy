from numpy import array, zeros, dot
from random import sample
from numpy.random import choice, rand
from math import exp, sqrt

def get_l1_ball(dim, radius):
    #get the vertices of l1 ball of a given radius
    final = zeros((2*dim, dim))
    for i in range(2*dim):
        final[i][i//2] = radius*(1-2*(i%2))
    return final

def partition(Z, Y, size):
    #partitionate the set into a subset of a given size, and the other the complement
    selected = choice(Z.shape[0], size, replace=False)
    selected.sort()
    non_selected = []
    j = 0
    for i in range(len(Z)):
        if j >= len(selected) or selected[j] != i:
            non_selected.append(i)
        else:
            j += 1
    non_selected = array(non_selected)
    non_selected.sort()
    return Z[selected], Y[selected], Z[non_selected], Y[non_selected]

def get_sample(Z, Y, size):
    selected = choice(Z.shape[0], size, replace=False)
    return Z[selected], Y[selected]

def sigmoid(z):
    return 1/(1+exp(-z))

def generate_data(x_star, size, max_cordinate, EPS):
    #create a dataset of given size using x_star, the probability of being incorrect class is EPS
    dim = len(x_star)
    
    Z = zeros((size, dim))
    Y = zeros(size)
    largo = 0
    
    while largo < size:
        z = 2*max_cordinate*rand(dim) - max_cordinate
        z[0] = 1
        pred = sigmoid(dot(x_star, z))
        if pred < EPS:
            Z[largo] = z
            Y[largo] = 0
            largo += 1
        if pred > 1 - EPS:
            Z[largo] = z
            Y[largo] = 1
            largo += 1
    
    return array(Y), array(Z)


def sparse(size, x):
    #create a vector of a given size filling the vector x with zeros at random places
    newx = [0 for _ in range(size)]
    selected = choice([i for i in range(size)], len(x), replace=False)
    for i in range(len(x)):
        newx[selected[i]] = x[i]
    return newx

def get_random_in_sphere(dim, radio):
    x = 2*radio*rand(dim) - radio
    norm = sqrt(sum(xi**2 for xi in x))
    x = (rand()/norm)*x
    return x
