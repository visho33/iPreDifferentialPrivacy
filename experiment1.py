from pandas import read_csv
from numpy import array, zeros
from sets_manager import get_l1_ball
from loss import LossFunction
from privatepolyhedralstochasticfrankwolfe import PolySFW

print("cargando data-frame...")
df = read_csv('data.csv')
print("guardando data-frame...")
Y = array(df["y"])
df.drop('id', inplace = True, axis = 1)
df.drop('y', inplace = True, axis = 1)
df_new = df.iloc[:, 0:2]
Z = array(df_new)
dim = len(Z[0]) + 1
radius = 1
print(f"obteniendo el conjunto de {dim} vertices...")
vertices = get_l1_ball(dim, radius)
f = LossFunction()
f.L0 = 8*radius
f.L1 = 2
epsilon = 2
delta = 0.01
x = zeros(dim)
print("creando el modelo...")
x_zero = array([(1-2*(i%2))/(dim) for i in range(dim)])
print(x_zero)
print("el error con el vector inicial es ", f.set_f(x_zero, Y, Z))
model1 = PolySFW(Z, Y, epsilon, delta, vertices, f, 2*radius, x_zero)
print("corriendo el modelo...")
model1.run()
print("se obtuvo el siguiente punto óptimo: ", model1.x_priv)
print(f.set_f(model1.x_priv, Y, Z))