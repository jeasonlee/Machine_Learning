from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


iris = datasets.load_iris()
# print(iris)
X = iris.data[:,[0,1,2,3]]
# print(X)
y = iris.target

for i in range(len(y)):
    if i >= 100:
        y[i] = 1
print (y)
# print(iris.data)
# print(iris.target)
# print(iris)
# print(X)
# 将原始数据集划分为“训练集”和“测试集”
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# test_size = 0.3 表示30%的样本数据作为测试集
# random_state = 1 随机数种子，在需要重复试验的时候，保证得到一组一样的随机数。
# stratify = y 按照y中的比例划分数据集，即测试集和训练集的种类比例保持和y的种类比例一样

sc = StandardScaler()
# 估算训练数据中的miu和sigma
sc.fit(X_train)
# 使用训练数据中的miu和sigma对数据进行标准化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
answer = clf.predict(X_train)
# y_train = y_train.reshape(-1)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

answer = clf.predict(X_test)
# y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print(np.mean(answer == y_test))

# import numpy as np
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from minepy import MINE
# from scipy.stats import pearsonr
# # SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
# print(SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target))
#
# def mic(x, y):
#      m = MINE()
#      m.compute_score(x, y)
#      return (m.mic(), 0.5)
#
# res = lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x: mic(x, Y),X.T))).T))
# skb = SelectKBest(res, k=3)
# print(skb.fit_transform(iris.data, iris.target))
import os
from graphviz import Source
from sklearn.tree import export_graphviz
IMAGES_PATH = os.path.join("./images")
export_graphviz(
        clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names,
        class_names=["1",'2'],
        rounded=True,
        filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
