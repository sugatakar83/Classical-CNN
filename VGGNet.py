import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.layers import BatchNormalization

# create train and test dataset
train_csv = pd.read_csv('fashion-mnist_train.csv')
test_csv = pd.read_csv('fashion-mnist_test.csv')

X = train_csv.iloc[:,1:] # reading the images
y = train_csv.iloc[:,0]  # reading the corresponding labels
# normalize the images
X_test_actual = test_csv.iloc[:,1:]
X = X.to_numpy().reshape(len(X), 28, 28,1).astype('float32')
X_test_actual = X_test_actual.to_numpy().reshape(len(X_test_actual), 28, 28, 1).astype('float32')
X = X/255
X_test_actual = X_test_actual/255
n_classes=10
y = to_categorical(y, n_classes)
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model building
model = Sequential()
# 1st convolution block
model.add(Conv2D(64, 3, activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
# 2nd convolution block
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
# 3rd convolution block
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
# 4th convolution block
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1)) # default stride is 2
model.add(BatchNormalization())
# 5th convolution block
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1)) # default stride is 2
model.add(BatchNormalization())
# creating the flatten layer
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# model compilation
# categorical_crossentropy is used as a loss function for multi-class classification model
# where there are two or more output labels. The output label is assigned
# one-hot category encoding value in form of 0s and 1
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# train the model
print("Fit model on training data")
m_train = model.fit(X_train,
                    y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test, y_test)
                   )
