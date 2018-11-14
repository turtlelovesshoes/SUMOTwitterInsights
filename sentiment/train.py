from random import shuffle
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
import pandas as pd
import nltk
import pickle

STOP_WORDS = set(nltk.corpus.stopwords.words('english')) 

def tokenize_tweet(tweet):
    # this will clean text, ignoring 
    # punctuation and splitting into
    # individual words
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[^\w+@\s+]')
    

    # text is the last element
    # here we clean the text via the
    # tokenizer defined above
    text = tokenizer.tokenize(tweet)
    return " ".join([i.lower() for i in text if i not in STOP_WORDS])

data = pd.read_csv("./sts_gold_tweet.csv", sep=";", encoding="utf-8").drop('id', 1)

# replace 4 with 'pos' and 0 as 'neg' in 'polarity' column
data['polarity'] = data.replace({4: 'pos', 0: 'neg'})

# # tokenize text
#data['tweet'] = map(tokenize_tweet, data.tweet)

# convert the data into a list
data = data[['tweet', 'polarity']].values.tolist()


shuffle(data)
L = len(data)
train_index = int(0.60 * L)

# split the data into a train and test data
train, test = data[:train_index], data[train_index: ]


cl = NaiveBayesClassifier(train)
print("N records:", L)
print("Train Accuracy:", cl.accuracy(train))
print("Test Accuract", cl.accuracy(test))


def get_other():
	TRAINPATH = "./training.1600000.processed.noemoticon.csv"
	# map raw tweet labels
	# to human-readable labels
	SENTIMENT_MAP = {
	    0: "neg",
	    2: "neut",
	    4: "pos"
	}
	data = (
	        pd
	        .read_csv(TRAINPATH, encoding="ISO-8859-1", header=None)
	        .sample(10000)  # take a sample for now
	    )
	data = data[data[0] != 2]
	data[0] = [SENTIMENT_MAP[i] for i in data[0]]
	#data[5] = map(tokenize_tweet, data[5])
	return data[[5, 0]].values.tolist()


data += get_other()
shuffle(data)
L = len(data)
train_index = int(0.80 * L)

# split the data into a train and test data
train, test = data[:train_index], data[train_index: ]
cl = NaiveBayesClassifier(train)
print("N records:", L)
print("Train Accuracy:", cl.accuracy(train))
print("Test Accuract", cl.accuracy(test))

pickled_classifier_file = open('sentiment_classifier_big.obj', 'wb')
pickle.dump(cl, pickled_classifier_file)
pickled_classifier_file.close()

