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
    num=-1
    for i in range(10):
        if tx[i]==1:
            num=i
            break
    R=-np.log(y[num])
    dz=np.zeros(10)
    dz[num]=-1/y[num]
    return R, dz
def softmax(x):

    t=np.exp(-x)
    return t/np.sum(t)

def softmax_dx(dx,y):
    # y=softmax(x)
    w=y*dx
    return y*np.sum(w)-w
def relu(x):
    for i in range(len(x)):
        if x[i]<0:
            x[i]=0
    return x
def relu_dx(x):
    dx=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<0:
            dx[i]=0
        else: dx[i]=1
    return dx
def loss_function_sq(Dx,y):
    delta=Dx-y
    return np.dot(delta,delta), delta
def loss(A, B, x, tx, return_grad=True):
    z3=linear_layer(A,x)
    #z23=relu(z3)
    z2=linear_layer(B,z3)
    k=max(z2.max(),abs(z2.min()))/100
    z15=z2/k

    z1=softmax(z15)

    R, dz1 = loss_function_cr(tx,z1)
    if not return_grad: return R
    dz15=softmax_dx(dz1,z1)
    dz2=dz15*k

    dB = linear_layer_dtheta(dz2, z3)
    dz3 = linear_layer_dx(dz2, B)
    dA = linear_layer_dtheta(dz3, x)
    return R, dA, dB
    # z3=linear_layer(A, x)
    # z25=relu(z3)
    # z2=linear_layer(B, z25)
    # k=max(z2.max(),abs(z2.min()))/100
    # z2/=k
    # z1=softmax(z2)
    # R,dz1=loss_function_sq(tx, z1)
    #
    #
    #
    # if not return_grad: return R
    # dz2=softmax_dx(dz1, z1)
    #
    # dB=linear_layer_dtheta(dz2, z25)
    # dz3=linear_layer_dx(dz2, B)
    # # dz3=relu_dx(z25)
    # dA=linear_layer_dtheta(dz3, x)
    # return R, dA, dB
def train_on_batch(A, B, batch, number_of_steps=1, step_size=0.001):
    batch_size=len(batch[0])
    history=[]
    for _ in range(number_of_steps):
        dA=np.zeros(A.shape);
        dB=np.zeros(B.shape);
        error=0
        i=0
        for x,y in zip(*batch):
            i+=1
            R,DA,DB=loss(A,B,x,y)
            dA+=DA
            dB+=DB
            error += R;
            if i%200==0:
                A-=step_size/200*dA
                B-=step_size/200*dB
                #print(i,"loss:", error/200)

                history.append(error / 200)
                error=0
                dA=0
                dB=0






        # A-=step_size/batch_size*dA
        # B-=step_size/batch_size*dB

    return np.sum(history)/len(history)
def train_network(A, B, y_train, x_train, x_test, y_test, test=not None, number_of_steps=50, debug=False):
    report_each=number_of_steps/50
    history=[]
    if not test is None:
        print("Initial error {}".format(test_network(A, B, x_test,y_test)))
    try:
        for n in range(number_of_steps):
            error=train_on_batch(A, B, (x_train,y_train))
            if debug: print(n,":",error[-1])
            if not test is None and n%report_each==0:

                # file = open("B 3 epoch "+str(n)+".txt", 'w')
                # for i in range(len(B)):
                #     for j in range(len(B[i])):
                #         file.write(str(B[i][j]) + ' ')
                #     file.write('\n')
                # file.close()
                if n%10==0 and n!=0:
                    file = open("./weights/B 3-0.txt", 'w')
                    for i in range(len(B)):
                        for j in range(len(B[i])):
                            file.write(str(B[i][j]) + ' ')
                        file.write('\n')
                    file.close()
                    file = open("./weights/A 3-0.txt", 'w')
                    for i in range(len(A)):
                        for j in range(len(A[i])):
                            file.write(str(A[i][j]) + ' ')
                        file.write('\n')
                    file.close()
                print("Epoch {}, generalization error {}".format(n, error))

            history.extend([error])

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


file=open("perceptronweightstrainA.txt")
A=file.readlines()
for i in range(len(A)):
    A[i]=A[i].split()
    for j in range(len(A[i])):
        A[i][j]=float(A[i][j])

file.close()
file=open("perceptronweightstrainB.txt")
B=file.readlines()
for i in range(len(B)):
    B[i]=B[i].split()
    for j in range(len(B[i])):
        B[i][j]=float(B[i][j])

file.close()
A=np.array(A)
B=np.array(B)

def neuralnetwork(x,A=A,B=B):



    maxim=0
    maxz=-1
    while(maxz!=1):
        z3 = linear_layer(A, x)
        z2 = linear_layer(B, z3)
        k = max(z2.max(), abs(z2.min())) / 100
        z15 = z2 / k

        z1 = softmax(z15)
        y=np_utils.to_categorical(1,10)
        for i in range(len(z1)):
            if z1[i]>maxim:
                maxim=z1[i]
                maxz=i
        R, DA, DB = loss(A, B, x, y)
        A -= 0.0006*DA
        B-=0.0006*DB
    file = open("perceptronweightstrainA.txt", 'w')
    for i in range(len(A)):
        for j in range(len(A[i])):
            file.write(str(A[i][j]) + ' ')
        file.write('\n')
    file.close()
    file = open("perceptronweightstrainB.txt", 'w')
    for i in range(len(B)):
        for j in range(len(B[i])):
            file.write(str(B[i][j]) + ' ')
        file.write('\n')
    file.close()
    return maxz



#loss(A,B,x_test[0],y_test[0])
# features1=800
# features2=10
# A=np.random.rand(features1,784)
# B=np.random.rand(10,features1)


# d={}
# for i in range(10):
#     d[i]=0
# k=0
# for i in range(len(x_test)):
#
#      if neuralnetwork(A,B,x_test[i])!=y_testreal[i]:
#
#          d[y_testreal[i]]+=1
#          k+=1
# print(d)
# print(k)
# A=np.random.rand(200,len(x_train[0]))
# B=np.random.rand(10,200)
if __name__=="__main__":

    history=train_network(A, B, y_train, x_train,x_test,y_test)