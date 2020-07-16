import scipy.io
import matplotlib.pyplot as plt
import numpy as np
a=scipy.io.loadmat('MsFEM.mat')
xs = a['Xs']
ys = a['Ys']
us = a['Us']

plt.scatter(xs,ys,s=10,c=us, cmap='rainbow',marker='.')
plt.show()

x=np.linspace(0,0.1,100)
y=np.linspace(0,0.1,100)
X,Y = np.meshgrid(x,y) 
P=1.8
eps = np.sqrt(2)/1000
a_inv=(2.+ P *np.sin(2*np.pi*X/eps))*(2.+ P *np.sin(2*np.pi*y/eps))
a=1/a_inv
plt.contourf(X,Y,a)
plt.show()
