'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.datasets import imdb

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline

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

batch = X_train.shape[0]
sentence_size = X_train.shape[1]
embed_size = X_train.shape[2]

#X_train = X_train.reshape((batch,1,sentence_size,embed_size))
#X_valid = X_valid.reshape((X_valid.shape[0],1,sentence_size,embed_size))

maxlen = sentence_size
batch_size = 32

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True),
                        input_shape=(sentence_size,embed_size)))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.8))
model.add(Dense(6, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = 5,
          validation_data = [X_valid, y_valid])
          
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline

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

print(model.evaluate(X_test, y_test, batch_size=64, verbose=0))

test_output = model.predict(X_test, batch_size=64, verbose=0)

def output_to_onehot (output):
    print (output.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i][j] == np.amax(output[i]):
                output[i][j] = 1
            else:
                output[i][j] = 0
                
    return output

np.savetxt('test_predictions.txt', output_to_onehot(test_output), fmt='%i')
