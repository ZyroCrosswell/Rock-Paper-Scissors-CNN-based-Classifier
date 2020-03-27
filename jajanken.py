import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from PIL import Image

REBUILD_DATA = False

DATADIR = "data/"
CATEGORIES = ["rock", "paper", "scissors"]
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
	        for img in os.listdir(path):
	            img_array = Image.open(os.path.join(path,img)).convert(mode = 'L')
	            img_array = img_array.resize((50,50))
	            training_data.append([np.array(img_array),np.array(np.eye(3)[CATEGORIES.index(category)])])
	np.random.shuffle(training_data)

if REBUILD_DATA:
	create_training_data()
training_data = np.load("training_data.npy", allow_pickle = True)

X = []
Y = []
for i in training_data:
    X.append(i[0])
    Y.append(i[1])

X = np.array(X).reshape(-1,50,50,1)  #1 is for grayscale the last one

train_x = tf.keras.utils.normalize(X, axis = 1)
Y = np.array(Y)
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
train_x = train_x[:-val_size]
train_y = Y[:-val_size]
test_x = train_x[-val_size:]
test_y = Y[-val_size:]

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation = "relu"))
model.add(Dense(3, activation = "softmax"))

model.compile(optimizer='adam',
             loss='mean_squared_error',
             metrics=['accuracy'])
model.fit(train_x, train_y, epochs = 5, validation_data = (test_x,test_y))
