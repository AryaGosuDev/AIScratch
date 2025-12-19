

import numpy as nmp
print ( "import nmp")
from sklearn.neural_network import MLPClassifier

print ( "Begin ")

nmp.random.seed(3743278)

x0 = nmp.random.random ( 50 ) - 0.3
y0 = nmp.random.random ( 50 ) +0.3
x1 = nmp.random.random ( 50 ) + 0.3
y1 = nmp.random.random ( 50 ) - 0.3
x = nmp.zeros( (100, 2) )
x[:50, 0] = x0
x[:50, 1] = y0
x[50:, 0] = x1
x[50:, 1] = y1
y = nmp.array ( [0]*50 + [1]*50)

idx = nmp.argsort ( nmp.random.random( 100 ))
x = x[idx]
y = y[idx]

x_train = x[:75]
x_test = x[75:]
y_train = y[:75]
y_test = y[75:]

print ( "Done")


