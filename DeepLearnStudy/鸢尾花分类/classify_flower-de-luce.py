from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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


def knn_iris_gscv():
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
    estimator = KNeighborsClassifier()

    # 加入网格搜索与交叉验证
    param_dict = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_train)

    # 5.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("对比真实值和预测值:\n",y_test == y_predict)

    # 方法2计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率为: ",score)

    print("最佳参数:\n",estimator.best_params_)
    print("最佳预估器:\n",estimator.best_estimator_)
    print("最佳结果:\n",estimator.best_score_)
    print("最佳交叉结果:\n",estimator.cv_results_)

    return None

if __name__ == "__main__":
    # knn_iris()
    knn_iris_gscv()