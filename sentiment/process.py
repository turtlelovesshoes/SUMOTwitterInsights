import pickle
import pandas as pd
from textblob.classifiers import NaiveBayesClassifier
import re
import sys
import json


# csv file originally named "Week 3 Social Firefox 63 Desktop - Sheet8.csv"
# TARGET_PATH should be changed to reflect your local filepath
TARGETPATH = "/Users/bmiroglio/Desktop/archive/week3social.csv"


def read_target_data():
    '''
    read in only english tagged (EN) tweets
    from the specific path as a pandas dataframe
    '''
    data = pd.read_csv(TARGETPATH, encoding="ISO-8859-1")

    # only english tweets
    en = data[["EN" in str(i) for i in data.tags]]
    return en


def get_sentiment(tweet):
    '''
    given a tweet (raw text)
    return (sentiment, probability pos, probability neg)
    '''
    result = cl.prob_classify(tweet)
    return (result.max(), result.prob("pos"), result.prob("neg"))


def parse_mif(line):
    '''
    parse raw text that results
    from calling show_informative_features
    '''
    try:
        (word, high, _, 
         low, _, hval, _, 
         lval) = re.sub(r'\bcontains\(\b|\b\) = True\b', '', line).split()
        pos = float(hval) if high == 'pos' else float(lval)
        neg = float(hval) if high == 'neg' else float(lval)
        return {"word": word, "pos":pos, "neg":neg, "high": hval}
    except Exception as e:
        # for garage words the regex fails i.e 'ð\x9f\x91\x8d'
        return {}
    

def prepare_ratios_for_chart(N=30, 
                             infile="./data/ratios.csv", 
                             outfile="./data/mif.json"):
    '''
    convert csv file into a json 
    structure suited for D3
    '''
    json_data = [
        {
            'key': "Positive",
            'values': []
        },
        {
            'key': "Negative",
            'values': []
        }
    ]
    for row in pd.read_csv(infile).head(N).iterrows():
        row = row[1]
        json_data[0]['values'].append({"label": row.word, "value": row.pct_pos})
        json_data[1]['values'].append({"label": row.word, "value": -row.pct_neg})

    with open(outfile, 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    print("Loading Model...")
    # load in pre-trained model
    cl = pickle.load(open('./models/sentiment_classifier_big.obj', 'rb'))

    # read in target data
    data = read_target_data()

    print("Predicting Sentiment...")
    # predict sentiment on data's text
    data[['pred', 'p_pos', 'p_neg']] = data.text.apply(lambda x: pd.Series(get_sentiment(x)))

    print("Calculating Ratios...")
    # use predicted sentiment to fit a dummy model
    # allowing us to get pos:neg ratios
    dummy_train = data[['text', 'pred']].values.tolist()
    dummy_cl = NaiveBayesClassifier(dummy_train)

    # shove the ratio structure into a file
    sys.stdout = open('./data/mif.txt', 'w')
    dummy_cl.show_informative_features(100)
    sys.stdout = sys.__stdout__

    print("Preparing Report...")
    # parse the raw ratios file and create dataframe
    with open("./data/mif.txt") as f:
        mif = f.read().split('\n')[1:]
        mif_df = pd.DataFrame([parse_mif(i) for i in mif])

    # create percentages from ratios
    N = (mif_df.neg + mif_df.pos)
    mif_df['pct_neg'] = mif_df.neg / N
    mif_df['pct_pos'] = mif_df.pos / N

    # save data sorted by the ratios
    mif_df.dropna().sort_values("high", ascending=False).head(100).to_csv("./data/ratios.csv", index=False)
    data.drop_duplicates().to_csv("./data/data_pred.csv", index=False)

    # create json of top N word ratios
    prepare_ratios_for_chart(N=100)

    # read in parsed json
    with open("./data/mif.json") as f:
        ratios_json = json.load(f)

    # edit file with inline json
    with open("./charts/barchart-horiz-template.html") as template:
        with open("./charts/barchart-horiz.html", "w") as report:
            for line in template:
                # inject json right into file
                if "%%DEFINE DATA HERE%%" in line:
                    
                    report.write("\t\tlet data = " + str(ratios_json))
                else:
                    report.write(line)