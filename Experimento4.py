from numpy import array
from sets_manager import get_l1_ball, generate_data, sparse
from loss import LossFunction
from polyhedralstochasticfrankwolfe import PolySFW
from numpy.random import rand

dimx = 8
radius = 1
radius_data = 5*dimx*dimx
x_star = 1/dimx*(radius*rand(dimx) - radius)
x_star = sparse(50, x_star)
print("generando data")
Y, Z = generate_data(x_star, 5000, radius_data, 0.05)
dim = len(Z[0]) + 1
print("encontrando los vertices")
vertices = get_l1_ball(dim, radius)
f = LossFunction()
f.L0 = 8*radius
f.L1 = 2
print("creando el modelo...")
x_zero = array([1/dim for _ in range(dim)])
print("el punto inicial es: ", x_zero)
print("el punto secreto es: ", x_star)
print("el error con el vector desconocido es ", f.set_f(x_star, Y, Z))
print("el error con el vector inicial es ", f.set_f(x_zero, Y, Z))
model1 = PolySFW(Z, Y, vertices, f, 2*radius, x_zero)
print("corriendo el modelo...")
model1.run()
print("se obtuvo el siguiente punto óptimo: ", model1.x_priv)
print("el error con el vector final es ", f.set_f(model1.x_priv, Y, Z))