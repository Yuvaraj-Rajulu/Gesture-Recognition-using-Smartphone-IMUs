# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 18:25:50 2018

@author: Yuvaraj
"""


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from os.path import basename
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import keras
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Embedding, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Bidirectional,GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

#os.environ['CUDA_VISIBLE_DEVICES']= '1'

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz"
]

# Output classes to learn how to classify
LABELS = [
    "Swipe_Left", 
    "Swipe_Right", 
    "Swipe_Up", 
    "Swipe_Down", 
    "Circle_CW", 
    "Circle_CCW",
    "Lower_Semi_Circle", 
    "Tilde", 
    "Infinity", 
    "Caret"
] 

DATASET_PATH = r'D:\UniStuttgart\Kurs\Sem2\HCILab\Spyder\data\dataset\\'


TRAIN = "train_OUT\\"
TEST = "test_OUT\\"

##########################################################################
# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
        
        print(len(X_signals))
    
    return np.transpose(np.array(X_signals), (1, 2, 0))
    #return np.array(X_signals)


X_train_signals_paths = [
    DATASET_PATH + TRAIN + "CHECK_" + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "CHECK_" + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
]


print(X_train_signals_paths)
print(X_test_signals_paths)

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

##############################################################

#print(X_train.shape)
#print(X_test.shape)

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "CHECK_y.txt"
y_test_path = DATASET_PATH + TEST + "CHECK_y.txt"

Y_train = load_y(y_train_path)
Y_test = load_y(y_test_path)


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

num_classes = 10

train_samples = 7000
test_samples = 1000

#X_test = X_train
#Y_test = Y_train

x_train, x_test, y_train, y_test = X_train[0:train_samples], X_test[0:test_samples], keras.utils.to_categorical(Y_train[0:train_samples], num_classes), keras.utils.to_categorical(Y_test[0:test_samples], num_classes)

#X_val = X_train[0:100]
#Y_val = Y_train[0:100]

timesteps = x_train[0].shape[0]
data_dim = x_train[0].shape[1]
batch_size = 50
epochs = 10

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

print(timesteps)
print(data_dim)
input_dim = data_dim

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, device_count = {'GPU' : 4})
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.9
#config.gpu_options.allocator_type = 'BFC'
with tf.device('/gpu:0'):
    model = Sequential()
    
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.5))
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
#     model.add(Dropout(0.5))
#     model.add(LSTM(32))  # return a single vector of dimension 32
#     model.add(Dropout(0.5))

#     model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    
    model.summary()
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
#              validation_split=0.1
              )
    score = model.evaluate(x_test, y_test, verbose=0)


Y_pred_class = model.predict(x_test)
Y_pred_class = np.argmax(Y_pred_class, axis=1)
print(Y_pred_class)

print(Y_test.shape)
print(Y_pred_class.shape)

#Y_pred_class = Y_pred_class.reshape(946,1)
Y_test = Y_test.reshape(946,)
Y_test = Y_test + 1
print(min(Y_test))
print(min(Y_pred_class))
print(Y_test.shape)
print(Y_pred_class.shape)


output = metrics.confusion_matrix(Y_test, Y_pred_class)
print(output)
output = np.roll(output, -1, axis=0)
print(output)
    
print('Test loss:', score[0])
print('Test accuracy:', score[1])



print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred_class)
confusion_matrix = np.roll(confusion_matrix, -1, axis=0)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()