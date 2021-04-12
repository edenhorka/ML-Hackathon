from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from textacy import preprocess


def clean_data(tweets):
    for i in range(len(tweets)):
        tweets[i] = preprocess.replace_urls(tweets[i], "")
        tweets[i] = preprocess.replace_numbers(tweets[i], "")
    return tweets


def classify(tweets_array):
    tweets_array = clean_data(tweets_array)
    with open("vec_file.pickle", 'rb') as vec_file:
        vec = pickle.load(vec_file)
        with open("LR_file.pickle", 'rb') as lr_file:
            lr = pickle.load(lr_file)
            tweets_array = vec.transform(tweets_array)
            return lr.predict(tweets_array)

