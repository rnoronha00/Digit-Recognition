##Roshan Noronha
##December 24, 2017
##This program trains a neural network that recognizes handwritten numbers

from keras.models import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import *
from keras.utils import *

#get the MNIST dataset
#split the dataset into training and testing datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plot the first 4 images
for i in range(1,5):
    plt.imshow(x_train[i])
    plt.show()

#print dimensions
#there are 60000, 28x28 matrices
#each 28x28 matrix is one handwritten number
#each handwritten number consists of 784 pixel values
print(x_train.shape)

#each matrix in the training/test needs to be "unrolled"
#unrolled values should be stored in a new matrix
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

#convert train and test outputs to one hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#create NN
#input layer is 784
#first hidden layer is 25 nodes. It uses relu.
#second hidden layer is 256 nodes. It uses relu.
#third hidden layer is 300 nodes. It uses relu.
#output layer is 10 nodes. It uses softmax to determine the probability of a class
model = Sequential()
model.add(Dense(25, input_dim = 28*28, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

#model.compile configures the learning process
#categorical_crossentropy defines the difference between the predicted output and expected output
#optimizer is gradient descent. adam is used for adaptive learning rates and Momentum
#metrics is set to ['accuracy']. This is done for any classification problem as per the Keras documentation
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

#train NN using model.fit
#epochs is the number of passes through the training data
#batch size is the number of samples that are sent through the network. In this case 128 samples are trained at one time. Networks train faster with mini batches since the weights update after each batch.
model.fit(x_train, y_train, epochs = 20, shuffle= True, verbose = 2, batch_size= 128)

#run NN on test data
test_error_rate = model.evaluate(x_test, y_test, verbose = 0)
print(model.metrics_names)
print(test_error_rate)

#save the NN so that it can be reused
model.save("trainedMNISTModel.h5")
print("NN saved to disk!")


