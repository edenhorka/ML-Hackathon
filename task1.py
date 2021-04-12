import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from classifier import clean_data
from collections import Counter
import matplotlib.pyplot as plt


paths = ["ConanOBrien_tweets.csv", "cristiano_tweets.csv", "donaldTrump_tweets.csv",
         "ellenShow_tweets.csv", "jimmykimmel_tweets.csv", "joeBiden_tweets.csv", "KimKardashian_tweets.csv",
         "labronJames_tweets.csv", "ladygaga_tweets.csv", "Schwarzenegger_tweets.csv"]


def most_common(path, n):
    tweets = pd.read_csv(path)["tweet"].values
    v = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, stop_words="english")
    summaries = "".join(clean_data(tweets))
    ngrams_summaries = v.build_analyzer()(summaries)
    return Counter(ngrams_summaries).most_common(n)


def organise_data():
    combined_csv = pd.concat([pd.read_csv(f) for f in paths], sort=False)
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')
    combined_csv = combined_csv.dropna()
    train_data, test_data = train_test_split(combined_csv, test_size=0.2)
    train_y = train_data["user"].values
    train_x = train_data["tweet"].values
    test_y = test_data["user"].values
    test_x = test_data["tweet"].values
    return clean_data(train_x), train_y, clean_data(test_x), test_y


def get_error_rate(predictions, y):
    return np.count_nonzero(predictions - y) / len(y)


# def is_english(x):
#     """
#     Detect whether or not the tweet is not in english
#     :param x:  a tweet
#     :return:  1 / 0
#     """
#     if x.isspace() or not x:
#         return 0
#     langid.set_languages(['ja', 'en', 'it', 'pt', 'es', 'zh'])
#     return int(langid.classify(x)[0] != 'en')
#
#
# features = [(lambda x: x.count('!')), (lambda x: x.count('#')), (lambda x: x.count('@')), (lambda x: sum(map(str.isupper, x))), (lambda x: x.count('$')), is_english]
#
#
# def add_features(X, clean_data):
#     f_matrix = []
#     for x in clean_data:
#         for f in features:
#             f_matrix.append(f(x))
#     f_matrix = np.array(f_matrix).reshape(X.shape[0], len(features))
#     new_matrix = np.insert(X, [X.shape[1]], f_matrix, 1)
#     return new_matrix


def main(d):
    train_x, train_y, test_x, test_y = organise_data()
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=d, lowercase=True, stop_words="english")
    X = vec.fit_transform(train_x)
    # X = add_features(X, train_x)
    lr = LogisticRegression(multi_class='multinomial', solver='saga')
    lr.fit(X, train_y)
    X_test = vec.transform(test_x)
    # X_test = add_features(X_test, test_x)
    test_prediction = lr.predict(X_test)
    error = get_error_rate(test_prediction,test_y)
    print("Logistic Regression error rate:")
    print(error)
    save_lr(lr)
    save_vec(vec)
    return error


def find_optimal_d():
    ds = [5000, 10000, 15000, 20000, 25000, 30000]
    accuracy = []
    for d in ds:
        acc = 0
        for i in range(10):
            acc+=1-main(d)
        accuracy.append(acc/10)
    plt.bar([str(d) for d in ds], accuracy, width=0.4, color='mediumspringgreen')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.xlabel('Number of features')
    plt.title('Classifier accuracy as a function of d-number of features')
    plt.show()


def save_lr(lr):
    with open('LR_file.pickle', 'wb') as open_file:
        pickle.dump(lr, open_file, pickle.HIGHEST_PROTOCOL)


def save_vec(vec):
    with open('vec_file.pickle', 'wb') as open_file:
        pickle.dump(vec, open_file, pickle.HIGHEST_PROTOCOL)


main(d=25000)


