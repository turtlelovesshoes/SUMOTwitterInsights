import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer

########################################
# Once you download the data
# and change the paths below,
# this script should run fine
# runnning the following commands 
# in a terminal from the top level
# directory.

# First, ensure all dependencies are installed
# pip install -r requirements.txt

# Run the script
# python3 sentiment/sentiment_example.py

#######################################

# csv file downloaded from
# https://www.kaggle.com/kazanova/sentiment140
# TRAINPATH should be changed to reflect your local 
# filepath
TRAINPATH = "/Users/bmiroglio/Desktop/archive/training.1600000.processed.noemoticon.csv"


# csv file originally named "Week 3 Social Firefox 63 Desktop - Sheet8.csv"
# TARGET_PATH should be changed to reflect your local filepath
TARGETPATH = "/Users/bmiroglio/Desktop/archive/week3social.csv"


# ignore pesky pandas warnings
pd.options.mode.chained_assignment = None 


# words to ignore i.e. "the", "a", ..., etc.
STOP_WORDS = set(nltk.corpus.stopwords.words('english')) 


sub


def tokenize_tweet(tweet):
    # this will clean text, ignoring 
    # punctuation and splitting into
    # individual words
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[^0-9a-zA-Z]+')
    
    # word stemmer, i.e. maximum -> maxim
    st = LancasterStemmer()

    # text is the last element
    # here we clean the text via the
    # tokenizer defined above
    text = tokenizer.tokenize(tweet)
    return [i.lower() for i in text if i not in STOP_WORDS]


def trim_data(data):
    # map 0, 2, 4 to neg, neut, pos
    # None for any non_training data
    if "label" in data:
        data.loc[:, "label"] = [SENTIMENT_MAP.get(x) for x in data["label"]]
    else:
        data.loc[:, "label"] = None

    # map each tweet to a list of non-stop words
    data.loc[:, "words"] = [tokenize_tweet(tweet) for tweet in data['text']]

    return data[["label", "words"]]


def read_train_data():
    data = (
        pd
        .read_csv(TRAINPATH, encoding="ISO-8859-1", header=None)
        .sample(10000)  # take a sample for now
    )

    # label the columns
    data.columns = ["label", "id", "date", 
                    "flag", "user", "text"]

    return trim_data(data)


def read_target_data():
    data = pd.read_csv(TARGETPATH, encoding="ISO-8859-1")

    # only english tweets
    en = data[["EN" in str(i) for i in data.tags]]
    return trim_data(en)
    

def get_top_words(df, top):
    df.reset_index(inplace=True)
    rows = []
    _ = df.apply(lambda row: [rows.append([row['label'], word]) 
                              for word in row.words], axis=1)
    df_new = pd.DataFrame(rows, columns=["label", "word"])
    return (
        df_new
        .groupby("word")
        .count()
        .reset_index()
        .sort_values("label", ascending=False)
        .head(top)
        .word
    )


def document_features(tweet_words, top_words):
    features = {}
    # if one of the n_most_common words appears
    # in tweet, mark it as "contains(word)"
    for word in top_words:
        features['contains({})'.format(word)] = (word in tweet_words)
    return features


if __name__ == "__main__":
    # This process takes ~1min and may cause your computer's
    # fan to work harder than usual :)


    print("Reading and Formatting Training Data...")
    tweets = read_train_data()


    # only use 10,000 entries for now
    # and only look at top 2000 words
    print("Constructing Training Classifier...")
    top_words = get_top_words(tweets, top=10000)
    tweet_words = [(i[1].words, i[1].label)for i in tweets.iterrows()]
    features = [(document_features(tweet, top_words), label)
                for (tweet, label) in tweet_words][:10000] # only use 10,000 entries for now



    # Use 90% for training, 10% for testing
    train_set = features[ : int(len(features) * .9) ]
    test_set = features[ int(len(features) * .9) : ]
    classifier = nltk.NaiveBayesClassifier.train(train_set)


    # Show accuracy of trained model
    print("Results:\n")
    print("Accuracy", nltk.classify.accuracy(classifier, test_set))


    # Now we can load the Firefox tweets to predict sentiment using
    # the above classifier
    print("Reading and Formatting Target Data...")
    firefox_tweets = read_target_data()
    firefox_tweet_words = [i[1].words for i in firefox_tweets.iterrows()]
    firefox_features = [document_features(tweet, top_words)
                        for tweet in firefox_tweet_words]
    # assign senitment label to firefox tweets                        
    firefox_tweets.loc[:, 'label'] = classifier.classify_many(firefox_features)

    # # We've now predicted sentiment for the Firefox tweets
    # # We can assume this as "ground truth" to create a 
    # # "dummy classifer" that allows us to identify words that
    # # most associated with negative and positive tweets
    # # this is almost an exact copy of the code under the print 
    # # statement "Constructing Training Classifier"
    print("Constructing Target (Dummy) Classifier...")
    ff_top_words = get_top_words(firefox_tweets, top=2000)
    ff_tweet_words = [(i[1].words, i[1].label)for i in firefox_tweets.iterrows()]
    ff_features = [(document_features(tweet, top_words), label)
                for (tweet, label) in ff_tweet_words]
    ff_classifier = nltk.NaiveBayesClassifier.train(ff_features)


    print("Most predictive words of sentiment in Firefox Tweets:\n")
    ff_classifier.show_most_informative_features(20)
