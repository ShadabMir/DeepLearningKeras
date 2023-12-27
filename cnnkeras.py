import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass
from mnistmlp import plot_results_mlp
SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 10
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 32
    NUM_CHANNELS: int = 3

@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS: int = 31
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.001

def cnn_model1(input_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation='relu',input_shape=input_shape))
    model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=3,padding="same",activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=512,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10,activation='softmax'))
    return model


model = cnn_model1()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

results = model.fit(x_train,y_train,batch_size=TrainingConfig.BATCH_SIZE,epochs=TrainingConfig.EPOCHS,verbose=1,validation_split=0.3)
training_loss = results.history["loss"]
training_accuracy = results.history["accuracy"]
validation_loss = results.history["val_loss"]
validation_accuracy = results.history["val_accuracy"]

plot_results_mlp([training_loss,validation_loss],ylabel="Loss",y_lim=[0.0,5.0],metric_name=["training loss","validation loss"],color=["b","g"])
plot_results_mlp([training_accuracy,validation_accuracy],ylabel="Accuracy",y_lim= [0.0,1.0],metric_name=["Training accuarcy","validation accuracy"],color=["b","g"])

test_loss, test_acc = model.evaluate(x_test,y_test)
prediction = model.predict(x_test)
prediction_labels = [np.argmax(i) for i in prediction]

y_test_labels = tf.argmax(y_test,axis=1)
conf_matrix = tf.math.confusion_matrix(y_test_labels,prediction_labels)
import seaborn as sn
sn.heatmap(conf_matrix,annot=True, fmt="d", annot_kws={"size": 12})
plt.show()




model.save("model_dropout")
