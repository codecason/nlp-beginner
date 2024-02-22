import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def train(data_arr, label_arr, n_class, iters=1000, alpha=0.1, lam=0.01):
    '''
    @description: softmax 训练函数
    @param {type} 
    @return: theta 参数
    '''
    # ++ n x f
    n_samples, n_features = data_arr.shape
    n_classes = n_class
    # 随机初始化权重矩阵, ++ 即theta
    weights = np.random.rand(n_class, n_features)
    # 定义损失结果
    all_loss = list()
    # 计算 one-hot 矩阵
    # ++ len(label_arr) == n_samples
    y_one_hot = one_hot(label_arr, n_samples, n_classes)
    for i in range(iters):
        # 计算 m * k 的分数矩阵
        # ++ data_arr is X (n, f), weights is (c, f)
        scores = np.dot(data_arr, weights.T)
        # 计算 softmax 的值
        # scores: n_sample * n_class
        # probs: n_sample * n_class
        probs = softmax(scores)
        assert probs.shape == (n_samples, n_class)
        # 计算损失函数值
        # ++ y_one_hot: n_sample * (n_class)
        # 1/m * sum(n_sample * log(p))
        loss = - (1.0 / n_samples) * np.sum(y_one_hot * np.log(probs))
        all_loss.append(loss)
        # 求解梯度
        dw = -(1.0 / n_samples) * \
            np.dot((y_one_hot - probs).T, data_arr) + lam * weights
        dw[:, 0] = dw[:, 0] - lam * weights[:, 0]
        # 更新权重矩阵
        weights = weights - alpha * dw
    return weights, all_loss


def softmax(scores):
    # 计算总和
    # ++ scores: (n, ) n is the dimension
    # out put: array<float>
    sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores) / sum_exp
    return softmax


def one_hot(label_arr, n_samples, n_classes):
    # n_samples: 10^4, label_arr: n_samples, n_classes: 5
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot


def predict(test_dataset, label_arr, weights):
    scores = np.dot(test_dataset, weights.T)
    probs = softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1, 1))


def load_dataset(path):
    # read ngram for short sentences
    def read_tsv(path, max_line=-1):
        with open(path) as f:
            header = f.readline().strip().split('\t')
            # header_2_idx
            h2map = {}
            for i, v in enumerate(header):
                h2map[v] = i

            if max_line < 0:
                max_line = -1
            lines = f.readlines()[:max_line]
            items = []
            for row in lines:
                # format: PhraseId	SentenceId	Phrase	Sentiment
                row = row.strip().split('\t')
                item = {'word_seg': row[2], 'class': int(row[3])}
                items.append(item)
        return pd.DataFrame(items)

    # train and test_set
    x_train = read_tsv(path, max_line=10000)
    y_train = x_train['class']

    # vocabulary, extract features
    x_train, x_test, y_train, y_test = train_test_split(
        x_train['word_seg'].tolist(), y_train, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
    x_train = vectorizer.fit_transform(x_train)

    # 不用fit, 因为没有label
    x_test = vectorizer.transform(x_test)
    print(x_train.shape)
    # xtrain is a sparse matrix
    return x_train.toarray(), y_train, x_test.toarray(), y_test


if __name__ == "__main__":
    # load the data features
    data_arr, label_arr, test_data_arr, test_label_arr = load_dataset('train.tsv')
    data_arr = np.array(data_arr)
    label_arr = np.array(label_arr).reshape((-1, 1))
    # 迭代次数? 学习率, 正则项
    weights, all_loss = train(data_arr, label_arr, n_class=5, lam=0.2)

    # 计算预测的准确率
    # test_data_arr, test_label_arr = load_test_dataset('test.tsv')
    test_data_arr = np.array(test_data_arr)
    test_label_arr = np.array(test_label_arr).reshape((-1, 1))
    n_test_samples = test_data_arr.shape[0]
    y_predict = predict(test_data_arr, test_label_arr, weights)
    accuray = np.sum(y_predict == test_label_arr) / n_test_samples
    print(accuray)
    # 准确率 0.3 for 10000 samples
    # 准确率 0.36 with lambda 0.2

    # 绘制损失函数
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1000), all_loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()
