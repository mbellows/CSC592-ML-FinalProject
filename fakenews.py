# Receive a Runtime Warning but it is only a warning that can be ignored and notebook continues to run without errors
import tensorflow as tf
import numpy as np
import math
import timeit
import theano

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras.utils import plot_model
import pydot

def get_embeddings():
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw embedding data
    X_train = np.load('./train_embeddings.npy')
    
    y_train = np.load('./train_labels.npy')
    
    X_valid = np.load('./valid_embeddings.npy')
    
    y_valid = np.load('./valid_labels.npy')
    
    X_test = np.load('./test_embeddings.npy')
    
    y_test = np.load('./test_labels.npy')

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

model = Sequential()
model.add(Conv2D(128, (2, 2), activation='relu', input_shape=(1,sentence_size,embed_size), dim_ordering='th'))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (4, 4), activation='relu'))

# Paper did not specify pool size
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Create graphs of our model (one without the shapes and one with the shapes)
plot_model(model, to_file='model.png')
plot_model(model, to_file='model_with_shapes.png', show_shapes=True)

# Train the model
history = model.fit(X_train, y_train, 
          batch_size=64, nb_epoch=10, verbose=1,validation_data=(X_valid,y_valid))

# Plot the model
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
