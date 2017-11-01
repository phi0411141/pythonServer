from __future__ import print_function
from flask import Flask, jsonify, request, json
from pymongo import MongoClient
from flask_restful import Api, Resource, abort
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.naive_bayes import BernoulliNB

filePath = 'data.txt'
app = Flask(__name__)


def pre():
    stopwords = nltk.corpus.stopwords.words('english')
    f = open(filePath, 'r', encoding='utf8')
    Q = []
    while (True):
        question = f.readline()
        if len(question) == 0: break
        cate = f.readline()
        Q.append((question, cate))
        # if (len(Q) >= 100): break
    return (stopwords, Q)


def makeDict(S, stopwords, notion):
    my_dict = set()
    result = []
    for s in S:
        V = makeBeauty(s, stopwords, notion)
        for v in V:
            if (v not in my_dict):
                my_dict.add(v)
                result.append(v)

    result = sorted(result)
    return result


def makeBeauty(s, stopwords, notion):
    v = nltk.tokenize.word_tokenize(s)
    result = []
    lmtzr = WordNetLemmatizer()
    for i in range(len(v)):
        v[i] = v[i].lower()
        if (v[i] not in stopwords) and (v[i] not in notion):
            temp = v[i]
            v[i] = lmtzr.lemmatize(v[i], 'v')
            if (v[i] == temp):
                v[i] = lmtzr.lemmatize(v[i], 'n')
            result.append(v[i])
    return result


def prepare():
    # make a set containing stopwords
    (stopwords, Q) = pre()
    # add ? !
    notion = set(string.printable)
    # read a file, and return (question, id)
    question = []
    for (q, c) in Q: question.append(q)
    dictionary = makeDict(question, stopwords, notion)
    return (dictionary, Q, notion, stopwords)


def convert(S, stopwords, notion, dictionary):
    v = makeBeauty(S, stopwords, notion)
    result = [0] * len(dictionary)
    for s in v:
        left = 0
        right = len(dictionary) - 1
        pos = 0
        while (left <= right):
            mid = int((left + right) / 2)
            argmid = dictionary[mid]
            if (argmid == s):
                pos = mid
                break
            else:
                if (argmid < s):
                    left = mid + 1
                else:
                    right = mid - 1

        result[pos] = 1
    return result


def split(X, y, n_categories):
    new_dict = {}
    n_examples = len(X)
    for c in range(n_categories): new_dict[c] = []
    for i in range(n_examples): new_dict[y[i]].append(X[i])

    result = {}
    cnt = -1
    for c in range(n_categories):
        if (len(new_dict[c]) > 10):
            cnt += 1
            result[cnt] = new_dict[c]
    return (cnt, result)


def train():
    (dictionary, Q, notion, stopwords) = prepare()

    X = []
    all_categories = []

    check = set()
    for (q, c) in Q:
        x = convert(q, notion, stopwords, dictionary)
        X.append(x)
        if (c not in check):
            all_categories.append(c)
            check.add(c)

    y = []
    for (q, c) in Q:
        index = all_categories.index(c)
        y.append(index)

    n_categories = len(all_categories)
    # split data
    (n_categories, new_dict) = split(X, y, n_categories)
    # ending

    P = {}
    for i in range(n_categories): P[i] = 0
    #
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    n_train = n_test = 0

    for c in range(n_categories):
        length = int(len(new_dict[c]) * 0.8)
        n_train += length
        n_test += len(new_dict[c]) - length
        X_train = X_train + new_dict[c][: length]
        y_train = y_train + [c] * length
        X_test = X_test + new_dict[c][length:]
        y_test = y_test + [c] * (len(new_dict[c]) - length)
        P[c] += length
    # ending divide

    clf = BernoulliNB()
    clf.fit(X, y)

    return (clf, stopwords, notion, dictionary, all_categories)


(clf, stopwords, notion, dictionary, categories) = train()


def test(s):
    global clf, stopwords, notion, dictionary, categories
    v = convert(s, stopwords, notion, dictionary)
    return categories[clf.predict([v])[0]]

@app.route('/')
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route('/', methods=['POST'])
def getdata():
    # client = MongoClient()
    # db = client.admin     
    body_unicode = request.data.decode('utf-8')
    dataString = json.loads(body_unicode)['string']   
    # db.aaa.insert_one({'body_unicode': json.loads(body_unicode)})
    return test(dataString)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
