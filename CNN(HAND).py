import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# LOADING DATASET FROM MNIST
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# NORMALIZING DATA BETWEEN 0 AND 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# RESHAPING DIMENSIONS AND RESIZING TO 28X28
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape((-1, IMG_SIZE, IMG_SIZE, 1))
x_testr = np.array(x_test).reshape((-1, IMG_SIZE, IMG_SIZE, 1))

print("Training Samples: ", x_trainr.shape)
print("Testing Samples: ", x_testr.shape)

# TRAINING ON 60000 MNIST DATASET
# CREATING A NEURAL NETWORK
model = Sequential()

# FIRST CONVOLUTION LAYER
model.add(Conv2D(64, (3, 3), input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))  # Activation function to make it non-linear
model.add(MaxPooling2D(pool_size=(2, 2)))  # MAX POOLING SINGLE MAX VALUE OF 2X2

# SECOND CONVOLUTION LAYER
model.add(Conv2D(64, (3,3), input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))  # Activation function to make it non-linear
model.add(MaxPooling2D(pool_size=(2, 2)))

# THIRD CONVOLUTION LAYER
model.add(Conv2D(64, (3,3), input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))  # Activation function to make it non-linear
model.add(MaxPooling2D(pool_size=(2, 2)))

# FULLY CONNECTED LAYER # 1
model.add(Flatten())  # NEEDS TO BE FLATTEN FROM 2D TO 1D
model.add(Dense(64))
model.add(Activation("relu"))

# FULLY CONNECTED LAYER # 2
model.add(Dense(32))
model.add(Activation("relu"))


# LAST FULLY CONNECTED LAYER
model.add(Dense(10))  # MUST BE EQUAL TO TOTAL NUMBER OF CLASSES HERE 10
model.add(Activation("softmax"))  # ACTIVATION FUNCTION WHICH SHOWS CLASS PROBABILITIES

# CHECKING SUMMARY OF MODEL
print(model.summary())

# COMPILING MODEL
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# FITTING ALGORITHM ON DATASET
model.fit(x_trainr, y_train, epochs=5, validation_split=0.3)

# EVALUATING MODEL ON TEST DATA
# FINDING LOSS AND ACCURACY
tLoss, tAccuracy = model.evaluate(x_testr, y_test)
print("Test Loss", tLoss*100)
print("Test Accuracy", tAccuracy*100)

# CHECKING PREDICTIONS
predictions = model.predict(x_testr)
# print(np.argmax(predictions[0]))
#
# plt.imshow(x_test[0])
# plt.show()

model.save('./HANDWRITTEN.h5')

