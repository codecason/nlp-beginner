from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import metrics

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import torch

## 
"""
x_train = pd.DataFrame
texts = x_train['word_seg'].tolist()  # list of text samples
labels = x_train['class'].tolist()  # list of corresponding labels (0 or 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
"""

"""
TODO: 分割数据集有哪些方法

"""

"""
x_train = read_tsv('./train.tsv', max_line)
texts = x_train['word_seg'].tolist()  # list of text samples
labels = x_train['class'].tolist()  # list of corresponding labels (0 or 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert the text to a bag-of-words representation
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test) # X_test is a sparse matrix
print(X_test.shape)
"""
