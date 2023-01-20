from pandas import read_csv
from numpy import array, zeros
from numpy.random import rand
from sets_manager import generate_data
from sets_manager import get_l1_ball
from loss import LossFunction
from privatevariancereducedfrankwolfe import PVarReducedFW

radius = 1
x_star = radius*rand(12) - radius
print("generando data")
Y, Z = generate_data(x_star, 1000, radius, 0.05)
dim = len(Z[0]) + 1
print("encontrando los vertices")
vertices = get_l1_ball(dim, radius)
f = LossFunction()
f.L0 = 8*radius
f.L1 = 2
epsilon = 2
print("creando el modelo...")
x_zero = array([(1-2*(i%2))/(dim) for i in range(dim)])
print(x_zero)
print("el error con el vector inicial es ", f.set_f(x_zero, Y, Z))
print("el error con el vector secreto es ", f.set_f(x_star, Y, Z))
model2 = PVarReducedFW(Z, Y, epsilon, vertices, f, 2*radius, x_zero, 10)
print("corriendo el modelo...")
model2.run()
print("se obtuvo el siguiente punto óptimo", model2.x_final)
print(f.set_f(model2.x_final, Y, Z))