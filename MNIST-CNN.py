# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import f1_score, accuracy_score

# Import data
# This dataset is available on Kaggle under the Digit Recognizer competition
test = pd.read_csv('../input/digit-recognizer/test.csv')
train = pd.read_csv('../input/digit-recognizer/train.csv')

y_train = train['label']
X_train = train.drop(labels = ['label'], axis=1)

# Tranforming data
X_train = X_train / 255
test = test / 255

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train, num_classes = 10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)

# The model
model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

batch_size = 32
epochs = 100

# Data augmentation
datagen = ImageDataGenerator(
          rotation_range=10,
          zoom_range = 0.1,
          width_shift_range=0.1,
          height_shift_range=0.1)

datagen.fit(X_train)

# Training the model
history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size),
                    epochs = epochs, 
                    validation_data = (X_val,y_val), 
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    verbose=2,
                    )

# Evaluating the model
history = pd.DataFrame(history.history)
print('Model Performance:')
print(('Objective: Accuracy: {:0.4f}')\
     .format(history['accuracy'].max()))
print(('Validation Accuracy: {:0.4f}')\
     .format(history['val_accuracy'].max()))
print(('Loss               : {:0.4f}')\
     .format(history['loss'].min()))
print(('Validation Loss    : {:0.4f}')\
     .format(history['val_loss'].min()))

# Thanks for reading!
