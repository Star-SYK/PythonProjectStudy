import jieba
import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def datasets_demo():
    """
    sklearn数据提取
    :return:
    """
    iris = load_iris()
    keys = iris.keys()

    # for key in keys:
    #     print(key)
    #     print(iris[key])
    #     print("\n")
    # print("鸢尾花数据集:\n",iris)
    # print("数据描述:\n",iris["DESCR"])
    x_train,x_test,y_train,y_test =train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("x_train:\n",x_train)
    print("x_test:\n",x_test)
    print("y_train:\n",y_train)
    print("y_test:\n",y_test)
    return None

def dict_Feture_extract():
    """
    字典特征提取:DictVectorizer
    :return:
    """
    data = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]

    # 1、实例化一个转换器
    transfer = DictVectorizer(sparse=False)

    # 2、调用fit_transfer
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new,type(data_new))
    print("特征名称:\n",transfer.get_feature_names_out())

    return None

def count_feture_extract():
    """
    文本特征提取:CountVectorizer
    :return:
    """
    data = ["the language, originally of England","now spoken in many other countries and used as a language of international communication throughout the world"]
    # data = ["干一行行一行","一行不行行行不行"]

    text_transfer = CountVectorizer(stop_words=["is","in","of"])

    data_new = text_transfer.fit_transform(data)

    print("data_new:\n",data_new.toarray())
    print("特征名称:\n", text_transfer.get_feature_names_out())
    return None
def cut_word(text):
    """
    使用jieba进行中文分词处理
    :param text:输入的带分词的中文字符串
    :return:
    """
    text = " ".join(list(jieba.cut(text)))
    return text

def count_chinese_demo():
    """
    中文文本特征提取，自动分词
    :return:
    """
    data = ["把学问过于用作装饰是虚假；完全依学问上的规则而断事是书生的怪癖",
            "学问是异常珍贵的东西，从任何源泉吸收都不可耻",
            "学到很多东西的诀窍，就是一下子不要学很多"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    text_transfer = CountVectorizer()

    data_final = text_transfer.fit_transform(data_new)

    print("data_new:\n",data_final.toarray())
    print("特征名称:\n", text_transfer.get_feature_names_out())

def tfidf_chinese_demo():
        """
        中文文本特征提取，自动分词
        :return:
        """
        data = ["把学问过于用作装饰是虚假；完全依学问上的规则而断事是书生的怪癖",
                "学问是异常珍贵的东西，从任何源泉吸收都不可耻",
                "学到很多东西的诀窍，就是一下子不要学很多"]

        data_new = []
        for sent in data:
            data_new.append(cut_word(sent))

        text_transfer = TfidfVectorizer(stop_words=["不要","就是"])

        data_final = text_transfer.fit_transform(data_new)

        print("data_new:\n", data_final.toarray())
        print("特征名称:\n", text_transfer.get_feature_names_out())

def data_normalization():
    """
    数据归一化处理
    :return:
    """
    # 1.导入数据
    data = pd.read_csv("dating.txt")

    print("data:\n",data)
    # 2.对数据进行归一化

    transfer = MinMaxScaler(feature_range=(2,3))

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

def data_stand():
    """
    数据标准化
    :return:
    """
    # 1.导入数据
    data = pd.read_csv("dating.txt")

    print("data:\n",data)
    # 2.对数据进行归一化

    transfer = StandardScaler()

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

def variance_demo():
    """
    过滤性方差特征
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("factor_returns.csv")
    print("data:\n",data)
    data = data.iloc[:, 1:-2]
    # 2.实例化一个转换器类
    transfer = VarianceThreshold(threshold=10)
    # 3.调用fit_transfrom
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new,data_new.shape)

    # 计算某两个变量之间的相关系数
    r = pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("相关系数:\n",r)

def pca_demo():
    """
    PCA降维
    :return:
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    # 1.实例化一个转换器类
    transfer = PCA(n_components=0.95)

    # 2.调用fit_transfrom
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)




if __name__ == '__main__':

    # 1.sklearn的数据
    # datasets_demo()

# 特征提取
    # 2.字典数据特征提取
    # dict_Feture_extract()

    # 3.文本数据特征提取
    # count_feture_extract()

    # 4.中文自动化分词并特征提取
    # count_chinese_demo()

#特征预处理
    # 5.tfidf进行特征提取
    # tfidf_chinese_demo()

    # 6.数据归一化处理
    # data_normalization()

    # 7.数据标准化
    # data_stand()

    # 8.数据特征值降维
    # variance_demo()

    # 9.pca降维
    pca_demo()