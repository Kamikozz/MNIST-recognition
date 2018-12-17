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
y_testreal=y_test
y_test=np_utils.to_categorical(y_test,10)
def linear_layer(theta, x):
    ans=np.dot(theta, x)
    return ans
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
    k=z2.max()
    z2/=k
    z1=softmax(z2)
    R,dz1=loss_function_cr(tx, z1)
    if not return_grad: return R
    dz2=softmax_dx(dz1, z1)

    dB=linear_layer_dtheta(dz2, z3)
    dz3=linear_layer_dx(dz2, B)
    dA=linear_layer_dtheta(dz3, x)
    return R, dA, dB
def train_on_batch(A, B, batch, number_of_steps=3, step_size=100):
    batch_size=len(batch[0])
    history=[]
    for _ in range(number_of_steps):
        dA=np.zeros(A.shape); dB=np.zeros(B.shape);
        error=0
        i=0
        for x,y in zip(*batch):
            i+=1
            R,DA,DB=loss(A,B,x,y)
            #step_size=A.max()/10
            A-=step_size/batch_size*DA
            B-=step_size/batch_size*DB
            error += R;
        A /= A.max()
        B /= B.max()



        # A-=step_size/batch_size*dA
        # B-=step_size/batch_size*dB
        history.append(error/batch_size)
    return history
def train_network(A, B, y_train, x_train, x_test, y_test, test=not None, number_of_steps=10, debug=False):
    report_each=number_of_steps/10
    history=[]
    if not test is None:
        print("Initial error {}".format(test_network(A, B, x_test,y_test)))
    try:
        for n in range(number_of_steps):
            error=train_on_batch(A, B, (x_train,y_train))
            if debug: print(n,":",error[-1])
            if not test is None and n%report_each==0:
                file = open("A 3 epoch "+str(n)+".txt", 'w')
                for i in range(len(A)):
                    for j in range(len(A[i])):
                        file.write(str(A[i][j]) + ' ')
                    file.write('\n')
                file.close()
                file = open("B 3 epoch "+str(n)+".txt", 'w')
                for i in range(len(B)):
                    for j in range(len(B[i])):
                        file.write(str(B[i][j]) + ' ')
                    file.write('\n')
                file.close()

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
        error+=loss(A,B,x,y,return_grad=False)
    return error/number_of_samples


file=open("A neuralweights.txt")
A=file.readlines()
for i in range(len(A)):
    A[i]=A[i].split()
    for j in range(len(A[i])):
        A[i][j]=float(A[i][j])

file.close()
file=open("B neuralweights.txt")
B=file.readlines()
for i in range(len(B)):
    B[i]=B[i].split()
    for j in range(len(B[i])):
        B[i][j]=float(B[i][j])

file.close()
A=np.array(A)
B=np.array(B)

def neuralnetwork(A,B,x):
    z3 = linear_layer(A, x)
    z2 = linear_layer(B, z3)
    k = z2.max()
    z2 /= k
    z1 = softmax(z2)
    max=0
    maxz=-1
    for i in range(len(z1)):
        if z1[i]>max:
            max=z1[i]
            maxz=i
    return maxz



#loss(A,B,x_test[0],y_test[0])
# features1=800
# features2=10
# A=np.random.rand(features1,784)
# B=np.random.rand(10,features1)


history=train_network(A, B, y_train, x_train,x_test,y_test)

d={}
for i in range(10):
    d[i]=0
k=0
for i in range(len(x_test)):

    if neuralnetwork(A,B,x_test[i])!=y_testreal[i]:

        d[y_testreal[i]]+=1
        k+=1
print(d)
print(k)
