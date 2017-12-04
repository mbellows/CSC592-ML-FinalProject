# Receive a Runtime Warning but it is only a warning that can be ignored and notebook continues to run without errors
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
%matplotlib inline

def get_embeddings():
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw embedding data
    X_train = np.load('embedding_output.npy')
    
    y_train = np.load('label_output.npy') 
    
    batch = X_train.shape[0]
    sentence_size = X_train.shape[1]
    embed_size = X_train.shape[2]
    
    X_train = tf.reshape(X_train, shape=[batch, 1, sentence_size, embed_size])

    #return X_train, y_train
    return X_train, y_train


# Invoke the above function to get our data.
X_train, y_train = get_embeddings()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
