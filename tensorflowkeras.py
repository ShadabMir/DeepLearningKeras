import keras.optimizers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Input ,Activation
import keras.datasets.boston_housing as bst
from keras import layers

seed_value = 42
np.random.seed(42)
tf.random.set_seed(seed_value)

(x_train,y_train),(x_test,y_test) = bst.load_data()
print(x_train.shape)

x_train_column = x_train[:,5]

plt.figure(figsize=[20,20])
plt.xlabel("Number of rooms")
plt.ylabel("median price $k")
plt.scatter(x_train_column,y_train,color="red",alpha=0.5)
plt.show()

model = Sequential()
model.add(Dense(units=1,input_shape=(1,))) # Initialize number of layers and parameters(weights and biases)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.005),loss="mse") # State the method of optimization(backprop) and learning rate + loss function used
history = model.fit(x_train_column,y_train,batch_size=16,epochs=101,validation_split=0.3) # Train the model using the data with epochs and a validation section
plt.figure(figsize=[20,20])
plt.plot(history.history['loss'],'g--',label="Training loss")
plt.plot(history.history['val_loss'],'bo',label="validation loss")
plt.xlim([0,100])
plt.ylim([0,300])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

x = [4,6,7,8,9]
y_pred = model.predict(x_test[:,5])
for i in range(len(x_test)):
    print("Predicted value of houss with ",x_test[:,5][i], " houses is ",y_pred[i] )