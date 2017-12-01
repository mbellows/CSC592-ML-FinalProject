import csv
import nltk
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Sammi/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

print ("Google News Word2Vec Model loaded successufully...")

embeddings = []

with open('./datasets/train.csv') as file:
    reader = csv.DictReader(file, delimiter=',')
    
    for row in reader:
        try:
            statements = nltk.word_tokenize(row['Statement'])
            embeddings = model[statements]
        except KeyError:
            print (statements, " not in vocabulary...")
            embeddings = "Unknown"
        
print ("Embedded statements from data set successufully...")

print (embeddings.shape)
print (embeddings)
