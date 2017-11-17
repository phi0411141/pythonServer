from __future__ import print_function
from flask import Flask, jsonify, request, json
from pymongo import MongoClient
from flask_restful import Api, Resource, abort
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import LinearSVC
import string
import numpy as np

app = Flask(__name__)
fileTrain = 'train.txt'


# fileTest = 'test.txt'

def readData(filePath):
    f = open(filePath, 'r', encoding='utf8')
    L = [line for line in f]
    questions = [L[i] for i in range(len(L)) if i % 2 == 0]
    categories = [L[i] for i in range(len(L)) if i % 2 == 1]
    data = {"X": questions, "y": categories}
    return data


def build_the_dictionary(questions):
    check = set()
    my_dict = []
    for question in questions:
        for v in build_the_list_tokenize(question):
            t = v
            if (v.isnumeric()): t = "100"
            if (t not in check):
                check.add(t)
                my_dict.append(t)

    return ["unknown"] + sorted(my_dict)


def build_the_list_tokenize(sentence):
    notion = ['?', "'s", "'", "`", "!", ",", ".", '"', '{', '}', '’', '“', '”']
    stopwords = nltk.corpus.stopwords.words('english') + notion
    lmtzr = WordNetLemmatizer()
    tokens = nltk.tokenize.word_tokenize(sentence)
    v = []
    for w in tokens:
        if (w.isupper()):
            v.append(w)
        else:
            v.append(w.lower())

    result = []
    for w in v:
        if (w not in stopwords):
            temp = w
            temp = lmtzr.lemmatize(temp, 'v')
            if (w == temp):
                temp = lmtzr.lemmatize(temp, 'n')
            result.append(temp)
    return result


def prepare():
    # read data
    data = readData(fileTrain)

    # build dictionary
    dictionary = build_the_dictionary(data["X"])

    return (data, dictionary)


def convert_to_vector(sentence, dictionary):
    result = [0] * len(dictionary)
    for s in build_the_list_tokenize(sentence):
        left = 1
        right = len(dictionary) - 1
        pos = 0
        while (left <= right):
            mid = int((left + right) / 2)
            X = dictionary[mid]
            if (X == s):
                pos = mid;
                break;
            else:
                if (X < s):
                    left = mid + 1
                else:
                    right = mid - 1
        result[pos] = 1
    return result


def classifier1(data, dictionary, categories):
    # whether business or not?
    # print("Building classifier 1: whether business or not...")
    X = [convert_to_vector(question, dictionary) for question in data["X"]]
    y = []
    for c in data["y"]:
        if c.startswith("Business"):
            y.append(categories.index(c))
        else:
            y.append(-1)

    clf = LinearSVC(random_state=0)
    clf.fit(X, y)
    return clf


def classifier2(data, dictionary, categories):
    # whether business or not?
    # print("Building classifier 2..")
    X = [convert_to_vector(question, dictionary) for question in data["X"]]
    y = []
    X_train = []

    for i in range(len(data["y"])):
        c = data["y"][i]
        if c.startswith("Business"): continue
        y.append(categories.index(c))
        X_train.append(X[i])

    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y)
    return clf


def main():
    # print("Making a dictionary...")

    (data, dictionary) = prepare()
    categories = list(set(data["y"]))
    # print(dictionary)

    clf1 = classifier1(data, dictionary, categories)
    clf2 = classifier2(data, dictionary, categories)
    return (clf1, clf2, dictionary, categories)


(clf1, clf2, dictionary, categories) = main()


def test(s):
    global clf1, clf2, dictionary, categories
    X = convert_to_vector(s, dictionary)
    pre_id = clf1.predict([X])[0]
    if (pre_id != -1):
        return categories[pre_id]
    else:
        new_pre_id = clf2.predict([X])[0]
        return categories[new_pre_id]


@app.route('/')
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"


@app.route('/', methods=['POST'])
def category_guess():
    # client = MongoClient()
    # db = client.admin
    data_string = None
    try:
        if request.method == 'POST':
            content = request.get_json(force=True)
            data_string = content.get('string')
    except ValueError:
        print("error parsing body:", ValueError)
        return jsonify(success=False, error='json error'), 400
    if data_string is None:
        return jsonify(success=False, error='cant get string from body'), 400
    guess_result = test(data_string)
    # db.aaa.insert_one({'body_unicode': json.loads(body_unicode)})
    return jsonify({'success': True, 'category_id': guess_result.strip('\n')}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')
