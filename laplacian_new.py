"""
@author: Wei Zhang
@Orignal DB_yw12.py
@data from pitzdaly of RANS
"""

#YW
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from itertools import product, combinations
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)

gpu_options = tf.GPUOptions(allow_growth=True)

ID = 0

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_uv, u, X_e, layers, lb, ub):
        self.i = 0 ##zW        

        self.lb = lb
        self.ub = ub
    
        self.x_uv = X_uv[:,0:1]
        self.y_uv = X_uv[:,1:2]
        
        self.x_e = X_e[:,0:1]
        self.y_e = X_e[:,1:2]
        
        self.layers = layers
        
        self.u = u
         
        # Initialize NNs
        self.weights, self.biases, self.W1, self.b1, self.W2, self.b2 = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True,
                                                     gpu_options=gpu_options))
        
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_uv_tf = tf.placeholder(tf.float32, shape=[None, self.x_uv.shape[1]])
        self.y_uv_tf = tf.placeholder(tf.float32, shape=[None, self.y_uv.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_e_tf = tf.placeholder(tf.float32, shape=[None, self.x_e.shape[1]])
        self.y_e_tf = tf.placeholder(tf.float32, shape=[None, self.y_e.shape[1]])

        self.f_pred  = self.net_NS(self.x_e_tf, self.y_e_tf)
        [self.u_pred, 
         self.a_pred] = self.net_v(self.x_uv_tf, self.y_uv_tf)
        
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) 
        self.loss_e = tf.reduce_mean(tf.square(self.f_pred))
               
        # switch between different definition of loss, like psi-omega, k, blabla    
        self.loss = self.loss_e + self.loss_u 
               
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        W1 = self.xavier_init(size=[layers[0], layers[1]])
        b1 = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)
  
        W2 = self.xavier_init(size=[layers[0], layers[1]])
        b2 = tf.Variable(tf.zeros([1,layers[1]], dtype=tf.float32), dtype=tf.float32)

        return weights, biases, W1, b1, W2, b2
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases, W1, b1, W2, b2):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        U = tf.tanh(tf.add(tf.matmul(H, W1), b1))
        V = tf.tanh(tf.add(tf.matmul(H, W2), b2))
          
        W = weights[0]
        b = biases[0]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))

        for l in range(1,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
   
   
    def net_v(self, x, y):
        res = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases, self.W1, self.b1, self.W2, self.b2)
        u = res[:,0:1]
        a = res[:,1:2]
        
        return u, a
        
    def net_NS(self, x, y):
        res = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases, self.W1, self.b1, self.W2, self.b2)
        u = res[:,0:1]
        a = res[:,1:2]
        
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        au_xx = tf.gradients(a*u_x, x)[0]
        au_yy = tf.gradients(a*u_y, y)[0]
       
        f_u = 1 + au_xx + au_yy
     
        return f_u
    
    def callback(self, loss, loss_u, loss_e):
        self.i += 1
        print('%dth, Loss: %.3e, Loss_u: %.3e, Loss_e: %3e' % (self.i, loss, loss_u, loss_e))

    def train(self, nIter, learning_rate):
        
        tf_dict = {self.x_uv_tf: self.x_uv, self.y_uv_tf: self.y_uv, self.u_tf: self.u,  
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e,
                   self.learning_rate: learning_rate}
     
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss = self.sess.run(self.loss, tf_dict)
                loss_u = self.sess.run(self.loss_u, tf_dict)
                loss_e = self.sess.run(self.loss_e, tf_dict)
                print('It: %d, Loss: %.3e, Loss_u: %.3e, Loss_e: %.3e Time: %.2f' % 
                      (it, loss, loss_u, loss_e, elapsed))
                start_time = time.time()
            
    
    def predict(self, X_star):
        
        tf_dict = {self.x_uv_tf: X_star[:,0:1], self.y_uv_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        a_star = self.sess.run(self.nut_pred, tf_dict)
    
        return u_star, a_star

if __name__ == "__main__":
   
    #flow parameters  
    Re = 5.e+5
    # machine learning parameters 
    noise = 0.0

    N_uv = 5000
    Ne = 5000

    layers = [2, 50, 50, 50, 50, 50, 50, 2]  #  (x,y) ->(u,a)

    data = scipy.io.loadmat('./Data/MsFEM.mat')
    u_star=data['Us']
    x_star=data['Xs']
    y_star=data['Ys']
        
    X_star = np.hstack((x_star.flatten()[:,None], y_star.flatten()[:,None]))
    
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    ##  locations & data for Datas
    # data
    u_0 = u_star.flatten()[:,None]

    idx = np.random.choice(X_star.shape[0], N_uv, replace=False)
    X_uv_train = X_star[idx, :]
    u_train =  u_0[idx,:]
    
    # eqns
    idx = np.random.choice(X_star.shape[0], Ne, replace=False)
    X_e_train = X_star[idx, :]

    model = PhysicsInformedNN(X_uv_train, u_train, X_e_train, layers, lb, ub)
    
    start_time = time.time()
    model.train(nIter=100000, learning_rate=1.e-3)
    model.train(nIter=100000, learning_rate=1.e-4)
    model.train(nIter=100000, learning_rate=1.e-5)
    model.train(nIter=100000, learning_rate=1.e-6)

    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    u_pred, a_pred = model.predict(X_star)
    u_pred = u_pred.reshape(1,u_star.size)

    scipy.io.savemat('./figures/laplacian_Adam'+ ID +'_%d_%s.mat' %(ID,time.strftime('%d_%m_%Y')),
                     {'u_pred':u_pred,
                      'a_pred':a_pred})   
 
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    fig, ax = plt.subplots()
    ax.axis('off')
    
    ####### Row 0: u_star(x,y) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-0.4, left=0.15, right=0.85, wspace=0.1)
    ax = plt.subplot(gs0[:, 0])
    
    h = ax.scatter(x_star,y_star,s=10,c=u_star, cmap='rainbow',marker='.')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.scatter(X_e_train_m[:,0:1], X_e_train_m[:,1:2], s = 0.3, marker='o', c='r')
    ax.scatter(X_uv_train[:,0:1], X_uv_train[:,1:2], s = 2, marker='+', c='b')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u_star(x,y)$', fontsize = 10)
    ####### Row 1: u_pred(x,y) ##################
    ax = plt.subplot(gs0[:, 1])
    
    h = ax.scatter(x_star,y_star,s=10,c=u_pred, cmap='rainbow',marker='.')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u_pred(x,x)$', fontsize = 10)
    ####### Row 0: a_star(x,y) ##################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.5-0.1, bottom=0.05, left=0.15, right=0.85, wspace=0.1)
    ax = plt.subplot(gs1[:, 0])
    
    x=np.linspace(0,0.1,100)
    y=np.linspace(0,0.1,100)
    X,Y = np.meshgrid(x,y) 
    P=1.8
    eps = np.sqrt(2)/1000
    a_inv=(2.+ P *np.sin(2*np.pi*X/eps))*(2.+ P *np.sin(2*np.pi*y/eps))
    a_star=1/a_inv

    h = ax.scatter(x_star,y_star,s=10,c=a_star, cmap='rainbow',marker='.')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$a_star(x,y)$', fontsize = 10)
    
    ####### Row 1: v_str(t,x) ##################
    ax = plt.subplot(gs1[:, 1])
    
    h = ax.scatter(x_star,y_star,s=10,c=a_pred, cmap='rainbow',marker='.')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$a_pred(x,y)$', fontsize = 10)
    
    plt.savefig('./figures/laplaciant_' + ID + '.png')
