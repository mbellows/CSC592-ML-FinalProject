import csv
import nltk
import gensim
import re
import numpy as np

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
def pad_sentences(sentences, padding_word = ""):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    
    return padded_sentences

'''
Load the data and obtain labels and statements for fake news classifier
'''
def load_data(some_file):
    
    # Open the file and read all labels 
    with open(some_file) as file:
        labels = [row["Label"] for row in csv.DictReader(file)]
    
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
    padded_statements = pad_sentences(statements)
    print ('Padding done')
    
    return labels, padded_statements
    

def main():    
    labels, padded_statements = load_data('./data_sets/sample.csv')
    
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    
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
    
    print(embeddings.shape)
    print (embeddings)

main()
