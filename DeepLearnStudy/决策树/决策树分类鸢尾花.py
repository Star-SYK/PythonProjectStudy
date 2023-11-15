from io import StringIO

import pydotplus
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz


def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=22)

    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法预估器
    estimator= KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    # 5.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("对比真实值和预测值:\n",y_test == y_predict)

    # 方法2计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率为: ",score)

    return None

def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3.决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")

    estimator.fit(x_train,y_train)

    # dot_data = StringIO()
    # file_name ="iris_out.dot";
    # export_graphviz(estimator,
    #                 out_file=file_name,
    #                 feature_names=iris.feature_names,
    #                 class_names=iris.target_names,
    #                 filled=True,
    #                 rounded=True,
    #                 special_characters=True)
    # with open(file_name) as f:
    #     dot_graph = (type(dot_data))(f.read())
    #     graph = pydotplus.graph_from_dot_data(dot_graph.getvalue())  # 决策树可视化
    #     print(dot_graph)
    #     graph.write_pdf('iris2.pdf')

    # 5.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("对比真实值和预测值:\n",y_test == y_predict)

    # 方法2计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率为: ",score)

    return None

if __name__ == '__main__':
    print("KNN算法：")
    knn_iris()

    print("=="*20)
    print("决策树算法：")
    decision_iris()
