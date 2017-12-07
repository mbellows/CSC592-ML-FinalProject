import csv
import nltk
import gensim
import re
import numpy as np
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count


"""
CITE: clean_str(string) function courtesy of https://mxnet.incubator.apache.org/tutorials/nlp/cnn.html
Tokenization/string cleaning for all datasets except for SST.
Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
"""
def clean_str(string):
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()


"""
CITE: pad_sentences(sentences, padding_word="") function courtesy of https://mxnet.incubator.apache.org/tutorials/nlp/cnn.html
Pads all sentences to the same length. The length is defined by the longest sentence.
Returns padded sentences.
"""
padding = []

def pad_sentences(training, sentences, padding_word = ""):
    
    sequence_length = max(len(x) for x in sentences)
    padding.append(sequence_length)
    
    
    if training == False:
        sequence_length = padding[0]
    
    
    padded_sentences = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    
    return padded_sentences


def load_labels(some_file):
    # Open the file and read all labels 
    with open(some_file) as file:
        labels = [row["Label"] for row in csv.DictReader(file)]
        
    return labels
    

'''
Load the data and obtain labels and statements for fake news classifier
'''
def load_data(some_file, training):
    
    # Open the file and read all statements 
    with open(some_file) as file:
        statements = [row["Statement"] for row in csv.DictReader(file)]
    
    # Strip the statements of whitespace characters in beginning and end
    statements = [s.strip() for s in statements]
    print('Strip done')
    
    # Clean the statements
    statements = [clean_str(to_be_cleaned) for to_be_cleaned in statements]
    print('Cleaning done')
    
    # Split statements by words
    statements = [s.split(" ") for s in statements]
    print('Split done')
    
    # Pad the statements to all be the same length
    padded_statements = pad_sentences(training, statements)
    print ('Padding done')
    
    return padded_statements


# Apply one hot encoding to the labels
def one_hot_encoding(y_data):
    
    lookup, y = np.unique(y_data, return_inverse=True)
    
    K = len(lookup)
    
    targets = np.array(y).reshape(-1)
    one_hot_targets = np.eye(K, dtype=int)[targets]
    
    return one_hot_targets


# Use word3vec to create embeddings for training, validation, and test files
def word2vec(padded_statements):
    
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Sammi/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    
    print ("Google News Word2Vec Model loaded successufully...")
    
    
    # Create embeddings 
    embeddings = []

    for i in range(len(padded_statements)):
        temp_sentence = padded_statements[i]
        
        embed_sentence = []
        
        for word in temp_sentence:
            
            try:
                embed_sentence.append(model[word])
            
            except KeyError:
                embed_sentence.append(np.random.uniform(-0.25, 0.25, 300))
            
        embeddings.append(embed_sentence)
    
    embeddings = np.array(embeddings)
    
    return embeddings
    

# Save embeddings to an output file
def save_data(embeddings, y_labels, embed_output_name, label_output_name):
    
    np.save(embed_output_name, embeddings)
    
    np.save(label_output_name, y_labels)
    
    print ('Saved to output complete.')
    

def main():
    
    padded_train_statements = load_data('./datasets/train.csv', True)
    tr_labels = load_labels('./datasets/train.csv')
    train_labels = one_hot_encoding(tr_labels)
    train_embeddings = word2vec(padded_train_statements)
    save_data(train_embeddings, train_labels, 'train_embeddings', 'train_labels')
    
    padded_valid_statements = load_data('./datasets/valid.csv', False)
    val_labels = load_labels('./datasets/valid.csv')
    valid_labels = one_hot_encoding(val_labels)
    valid_embeddings = word2vec(padded_valid_statements)
    save_data(valid_embeddings, valid_labels, 'valid_embeddings', 'valid_labels')
    
    padded_test_statements = load_data('./datasets/test.csv', False)
    ts_labels = load_labels('./datasets/test.csv')
    test_labels = one_hot_encoding(ts_labels)
    test_embeddings = word2vec(padded_test_statements)
    save_data(test_embeddings, test_labels, 'test_embeddings', 'test_labels')
    
    
    
    
main()
