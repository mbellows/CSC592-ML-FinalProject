# Receive a Runtime Warning but it is only a warning that can be ignored and notebook continues to run without errors
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import theano

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

def get_embeddings():
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    folder = '/mnt/data/playground/temp-project-michael/'
    # Load the raw embedding data
    X_train = np.load(folder+'train_embeddings.npy')
    
    y_train = np.load(folder+'train_labels.npy')
    
    X_valid = np.load(folder+'valid_embeddings.npy')
    
    y_valid = np.load(folder+'valid_labels.npy')
    
    X_test = np.load(folder+'test_embeddings.npy')
    
    y_test = np.load(folder+'test_labels.npy')

    #return X_train, y_train
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_valid, y_valid, X_test, y_test = get_embeddings()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)

print('Validation data shape: ', X_valid.shape)
print('Validation labels shape: ', y_valid.shape)

print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

np.random.seed(123)

batch = X_train.shape[0]
sentence_size = X_train.shape[1]
embed_size = X_train.shape[2]

X_train = X_train.reshape((batch,1,sentence_size,embed_size))
X_valid = X_valid.reshape((X_valid.shape[0],1,sentence_size,embed_size))
print (X_train.shape)


X_train = X_train.astype('float32')
print (X_train.shape)
print (X_train.dtype)

model = Sequential()
model.add(Conv2D(128, (2, 2), activation='relu', input_shape=(1,sentence_size,embed_size), dim_ordering='th'))
print (model.output_shape)


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (4, 4), activation='relu'))

# Paper did not specify pool size
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.8))

print (model.output_shape)

model.add(Flatten())
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          batch_size=64, nb_epoch=10, verbose=1,validation_data=(X_valid,y_valid))
