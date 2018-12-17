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
model=Sequential()
model.add(Dense(800, input_dim=784, activation="relu",kernel_initializer="normal"))
model.add(Dense(10,activation="softmax",kernel_initializer="normal"))
model.compile(loss="categorical_crossentropy", optimizer="SGD",metrics=["accuracy"])
print(model.summary())
model.fit(x_train,y_train,batch_size=200,nb_epoch=20,verbose=1)