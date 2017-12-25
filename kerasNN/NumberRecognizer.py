from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

#load the model
model = load_model("trainedMNISTModel.h5")

#get the test MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plots each handwritten number
#the y-axis has the predicted value
#the x-axis has the actual value
for i in range(0, 10):
    prediction = model.predict(x_test[i].reshape(-1, 28 * 28))
    plt.imshow(x_test[i])
    plt.ylabel("Predicted Value: " + str(np.argmax(prediction)))
    plt.xlabel("Actual Value: " + str(y_test[i]))
    plt.show()

