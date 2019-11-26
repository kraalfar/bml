import numpy as np
import matplotlib.pyplot as plt
import pandas
import heapq
import sys

sys.setrecursionlimit(4700)


def read_cancer_dataset(path_to_csv):
    dtf = pandas.read_csv(path_to_csv).sample(frac=1)
    return np.array(dtf.values[:, 1:]).astype(float), np.array(dtf["label"] == "M").astype(int)


def read_spam_dataset(path_to_csv):
    dtf = pandas.read_csv(path_to_csv).sample(frac=1)
    return np.array(dtf.values[:, 1:]).astype(float), np.array(dtf["label"]).astype(int)


def train_test_split(X, y, ratio):
    l = round(ratio * len(X))
    return X[:l], y[:l], X[l:], y[l:]


def get_precision_recall_accuracy(y_pred, y_true):
    res = np.zeros((2, 2))
    for i in range(len(y_pred)):
        res[1 - y_pred[i], 1 - y_true[i]] += 1
    pre = [res[0, 0] / (res[0, 0] + res[0, 1]), res[1, 1] / (res[1, 1] + res[1, 0])]
    rec = [res[0, 0] / (res[0, 0] + res[1, 0]), res[1, 1] / (res[1, 1] + res[0, 1])]
    return np.array(pre), np.array(rec), (res[0, 0] + res[1, 1]) / len(y_pred)


def plot_precision_recall(X_train, y_train, X_test, y_test, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(np.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(ys) - 0.01, np.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)


def plot_roc_curve(X_train, y_train, X_test, y_test, max_k=30):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k in ks:
        colors.append([k / ks[-1], 0, 1 - k / ks[-1]])
        knearest = KNearest(k)
        knearest.fit(X_train, y_train)
        p_pred = [p[0] for p in knearest.predict_proba(X_test)]
        tpr = []
        fpr = []
        for w in np.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > w else 1) for p in p_pred]
            tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
            fpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize=(7, 7))
    for tpr, fpr, c in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=c)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


class PQueue:
    def __init__(self, max_len):
        self.lim = max_len
        self.queue = []

    def peek(self):
        if len(self.queue):
            top = heapq.heappop(self.queue)
            heapq.heappush(self.queue, top)
            return -top[0], top[1]
        else:
            return np.inf, -1

    def add(self, dist, index):
        if len(self.queue) < self.lim:
            heapq.heappush(self.queue, (-dist, index))

    def pop(self):
        if len(self.queue):
            top = heapq.heappop(self.queue)
            return -top[0], top[1]
        else:
            return np.inf, -1

    def array(self):
        res = []
        while len(self.queue):
            top = heapq.heappop(self.queue)
            res.append(top[1])
        return np.array(res[::-1])

    def ok(self):
        return len(self.queue) < self.lim


class KDTree:
    def __init__(self, X, leaf_size=40):
        self.data = np.asarray(X)
        self.n, self.m = self.data.shape
        self.ls = leaf_size
        self.tree = self.build_tree(np.arange(self.n), 0)

    class Node:
        def __init__(self, val, depth):
            self.plane = val
            self.depth = depth
            self.left_son = None
            self.right_son = None

    class Leaf(Node):
        def __init__(self, ind, depth):
            self.ind = ind
            super().__init__(None, depth)

    def split(self, ind, med, ax):
        return ind[self.data[ind, ax] <= med], ind[self.data[ind, ax] > med]

    def build_tree(self, ind, cur_d):
        if len(ind) <= self.ls:
            return KDTree.Leaf(ind, cur_d)
        ax = cur_d % self.m
        med = np.median(self.data[ind, ax])
        left, right = self.split(ind, med, ax)
        cur_node = KDTree.Node(med, cur_d)
        cur_node.left_son = self.build_tree(left, cur_d + 1)
        cur_node.right_son = self.build_tree(right, cur_d + 1)
        return cur_node

    def query1(self, x, k=1):
        q = PQueue(k)
        self._query1(x, q, self.tree)
        return q.array()

    def _query1(self, x, q, node):
        if node is None:
            return
        if node.plane is None:
            for i in node.ind:
                cur = np.linalg.norm(x - self.data[i])
                if q.ok():
                    q.add(cur, i)
                else:
                    top = q.pop()
                    if top[0] > cur:
                        q.add(cur, i)
                    else:
                        q.add(top[0], top[1])
        else:
            ax = node.depth % self.m
            if x[ax] <= node.plane:
                self._query1(x, q, node.left_son)
                top = q.peek()
                if top[0] > node.plane - x[ax]:
                    self._query1(x, q, node.right_son)
            else:
                self._query1(x, q, node.right_son)
                top = q.peek()
                if top[0] > x[ax] - node.plane:
                    self._query1(x, q, node.left_son)

    def query(self, X, k=1):
        res = np.zeros((X.shape[0], min(k, self.n)))
        for i, x in enumerate(X):
            res[i] = self.query1(x, k).astype(int)
        return res


class KNearest:
    def __init__(self, n_neighbors=5, leaf_size=30):
        self.leaf_size = leaf_size
        self.k = n_neighbors
        self.tree = None
        self.y = None

    def fit(self, X, y):
        self.tree = KDTree(X, self.leaf_size)
        self.y = y

    def one_prob(self, x):
        near = self.tree.query1(x, self.k)
        res = sum(self.y[near]) / self.k
        return np.array([1 - res, res])

    def predict_proba(self, X):
        res = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            res[i] = self.one_prob(x)
        return res

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# X, y = read_cancer_dataset("cancer.csv")
# X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
# plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
# plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)
#


