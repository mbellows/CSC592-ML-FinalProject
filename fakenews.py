# Receive a Runtime Warning but it is only a warning that can be ignored and notebook continues to run without errors
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
#matplotlib inline

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


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

def my_model(X,y,is_training,filterSize,numFilters):

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=X, filters=numFilters, kernel_size=filterSize, padding="same", activation=tf.nn.relu)
    #Batch normalization #1
    batchnorm1 = tf.layers.batch_normalization(inputs=conv1, training=is_training)
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(inputs=batchnorm1, filters=(numFilters*2), kernel_size=filterSize, padding="same", activation=tf.nn.relu)
    #Batch normalization #2
    batchnorm2 = tf.layers.batch_normalization(inputs=conv2, training=is_training)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=batchnorm2, pool_size=[2, 2], strides=2)

    #Dense layer with batch norm
    pool1Size = np.asarray(pool1.get_shape().as_list()[1:]).prod()
    flat1 = tf.reshape(pool1,[-1,pool1Size])
    dense = tf.layers.dense(inputs=flat1, units=1024, activation=tf.nn.relu)
    batchnorm3 = tf.layers.batch_normalization(dense, training=is_training)

    #dropout
    dropout = tf.layers.dropout(inputs=batchnorm3, rate=0.4, training=is_training)

    #Output layer
    y_out = tf.layers.dense(inputs=dropout, units=10)

    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

sess = tf.Session()

y_out = my_model(X,y,is_training,filterSize=5,numFilters=26)

#DO! -- For testing different loss functions and optimizers
softmax = tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=tf.one_hot(y,10),logits=y_out,dim=-1,name=None)
mean_loss = tf.reduce_mean(softmax,axis=None,keep_dims=False,name=None,reduction_indices=None)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)


sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
