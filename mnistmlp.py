import keras.optimizers
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,FormatStrFormatter)
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.datasets import mnist,fashion_mnist
from keras.utils import to_categorical
from keras import layers

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# Loading the dataset - you can do this through pandas for your own datasets
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Split training data into validation and training set
x_validation = x_train[:10000]
x_train = x_train[10000:]

y_valid = y_train[:10000]
y_train = y_train[10000:]

# To be input into this neural network we have to reshape the images from a 2D array to a 1D array using reshape then noramlizing the values
x_train = x_train.reshape(x_train.shape[0],(x_train.shape[1] * x_train.shape[2]))
x_train = x_train.astype("float32")/255

x_test = x_test.reshape(x_test.shape[0],(x_test.shape[1] * x_test.shape[2]))
x_test = x_test.astype("float32")/255

x_validation = x_validation.reshape(x_validation.shape[0],(x_validation.shape[1] * x_validation.shape[2]))
x_validation = x_validation.astype("float32") / 255

# One hot encoding for catgorical variables(not actually needed just for practice)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_valid = to_categorical(y_valid)

# Creating the neural network architecture
model = tf.keras.Sequential()
"""The neural network has 2 hidden layers and one output layer
We only declare the hidden and output layers in tensorflow and the activation function used for their output"""
model.add(Dense(128,activation="relu",input_shape=(x_train.shape[1],)))# size of image input to the network
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax')) # softmax is used to normalize the ouputs in the output layer so that they are probabilities between 0 and 1


"""For one hot encoded categorical variables - use categoric cross entropy
for integer coded categoric variables - use sparse_categorical_crossentropy
For binary classification use binary_crossentropy"""
model.compile(optimizer='rmsprop',loss="categorical_crossentropy",metrics=["accuracy"])

results = model.fit(x_train,y_train,epochs=21,batch_size=64,validation_data=(x_validation,y_valid))

def plot_results_mlp(metrics,title=None,ylabel=None,y_lim=None,metric_name = None,color = None):
    fig , ax = plt.subplots(figsize=[15,4])
    if not(isinstance(metric_name,list) or (isinstance(metric_name,tuple))):
        metrics = [metrics,]
        metric_name = [metric_name,]
    for idx,metric in enumerate(metrics):
        ax.plot(metric,color=color[idx])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0,20])
    plt.ylim(y_lim)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.legend(metric_name)
    plt.show()
    plt.close()

training_loss = results.history["loss"]
training_accuracy = results.history["accuracy"]
valid_loss = results.history["val_loss"]
valid_accuracy = results.history["val_accuracy"]
plot_results_mlp([training_loss,valid_loss],ylabel="Loss",y_lim=[0.0,0.5],metric_name=["Training loss,Validation Loss"],color=["g","b"])
plot_results_mlp([training_accuracy,valid_accuracy],ylabel="Accuracy",y_lim=[0.9,1.0],metric_name = ["Training Accuracy, validation Accuracy"],color=["g","b"])


predictions = model.predict(x_test)
print("The ground truth is ",y_test[0])
for i in range(10):
    print("the probability of the digit ", i, " is ", predictions[0][i])

predicted_labels = [np.argmax(i) for i in predictions]
y_test_integer_labels = tf.argmax(y_test,axis=1)

conf_matrix = tf.math.confusion_matrix(labels=y_test_integer_labels,predictions=predicted_labels)
plt.figure(figsize=[20,5])
import seaborn as sn
sn.heatmap(conf_matrix,annot=True,fmt='d',annot_kws={"size":14})
plt.show()