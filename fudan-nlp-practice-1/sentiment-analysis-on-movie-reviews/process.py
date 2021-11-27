# encoding: -utf-8-
import enum
import os
import json
import logging
import numpy
import re
from sklearn.feature_extraction.text import CountVectorizer


def debug_dump(prefix, obj):
    name = prefix + '.json'
    with open(name, 'w') as f:
        f.write(json.dumps(obj, indent=4))
    logging.info(f"dump to {name}")


# implement a word bag model
class State:
    NEGATIVE = 0
    SOME_NEGATIVE = 1
    NEUTRAL = 2
    SOME_POSITIVE = 3
    POSITIVE = 4


class ModelTrain(object):
    def shuffle(self):
        pass

    def calculate_accuracy(self):
        pass


class Tokenizer(object):
    def bag_of_word(self, lines):
        bag = {}
        for line in lines:
            token_list = self.tokenize(line)
            for token in token_list:
                if token not in bag:
                    bag[token] = 1
                else:
                    bag[token] += 1
        features = self._set_bag_feature(bag)
        matrix = []
        for line in lines:
            statis = [0] * len(features)
            for word in line:
                kid = features.get(word, 0)
                statis[kid] += 1
            matrix.append(statis)
        return bag, matrix, features

    def tokenize(self, line):
        """
        really slow implementation
        """
        res = ''
        for c in line:
            if c in (',', ':', '\'', ';'):
                continue
            res += c
        return res.split(' ')

    def _set_bag_feature(self, bag):
        features = {}
        keys = sorted(bag.keys())
        for i, key in enumerate(keys):
            features[key] = i
        return features

    def expect_test(self, segments: list):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(segments)
        return X


def read_train():
    path = os.path.abspath(
        os.path.join(os.path.join(__file__, '../train.tsv'), 'train.tsv'))
    values = []
    with open(path, 'r') as f:
        values = f.readlines()
    # # strip first line
    values = values[1:10]
    segments = []
    for line in values:
        line = line.strip().split('\t')
        segments.append(line[2])

    print(json.dumps(segments, indent=4))
    # N-gram 词袋模型
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(segments)
    print(vectorizer.get_feature_names())
    print(X.toarray())


if __name__ == '__main__':
    read_train()
