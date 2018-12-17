from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
np.random.seed(42);
(x_train, y_train),(x_test, y_test)=mnist.load_data()
x_train=x_train.reshape(60000,784).astype('float32')
x_train/=255
x_test=x_test.reshape(10000,784).astype('float32')
x_test/=255
y_train=np_utils.to_categorical(y_train,10)
def linear_layer(theta, x):
    return np.dot(theta, x)
def linear_layer_dx(dx, theta):
    return np.dot(dx, theta)

def linear_layer_dtheta(dx, x):
    return dx[:,None]*x[None,:]
def loss_function_cr(tx,y):
    return -np.sum(tx*np.log(y)), -tx/y
def softmax(x):
    t=np.exp(-x)
    return t/np.sum(t)

def softmax_dx(dx,y):
    # y=softmax(x)
    w=y*dx
    return y*np.sum(w)-w
def loss(A, B, x, tx, return_grad=True):
    z3=linear_layer(A, x)
    z2=linear_layer(B, z3)
    z1=softmax(z2)
    R,dz1=loss_function_cr(tx, z1)
    if not return_grad: return R
    dz2=softmax_dx(dz1, z1)
    dB=linear_layer_dtheta(dz2, z3)
    dz3=linear_layer_dx(dz2, B)
    dA=linear_layer_dtheta(dz3, x)
    return R, dA, dB
def train_on_batch(A, B, batch, number_of_steps=3, step_size=5):
    batch_size=len(batch[0])
    history=[]
    for _ in range(number_of_steps):
        dA=np.zeros(A.shape); dB=np.zeros(B.shape);
        error=0
        for y,x in zip(*batch):

            R,DA,DB=loss(A,B,y,x)
            error+=R; dA+=DA; dB+=DB
        A-=step_size/batch_size*dA
        B-=step_size/batch_size*dB
        history.append(error/batch_size)
    return history
def train_network(A, B, y_train, x_train, x_test, y_test, test=None, number_of_steps=1000, debug=False):
    report_each=number_of_steps/10
    history=[]
    if not test is None:
        print("Initial error {}".format(test_network(A, B, x_test,y_test)))
    try:
        for n in range(number_of_steps):
            error=train_on_batch(A, B, (y_train,x_train))
            if debug: print(n,":",error[-1])
            if not test is None and n%report_each==0:
                print("Epoch {}, generalization error {}".format(n, test_network(A, B, x_test,y_test)))
            history.extend(error)
    except KeyboardInterrupt:
        pass
    return history
def test_network(A, B, x_test,y_test, number_of_samples=10000):
    error=0
    for i in range(number_of_samples):
        y=y_test[i]
        x=x_test[i]
        error+=loss(A,B,y,x,return_grad=False)
    return error/number_of_samples