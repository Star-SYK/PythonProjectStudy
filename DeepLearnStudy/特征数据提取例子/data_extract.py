import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import k_means


def data_processing():
    # 1.获取数据
    aisles = pd.read_csv("./data/aisles.csv")
    orders = pd.read_csv("./data/orders.csv")
    products = pd.read_csv("./data/products.csv")
    order_products__prior = pd.read_csv("./data/order_products__prior.csv")

    # 2、合并表
    # order_products_prior.csv:订单与商品信息
        # 字段: order_id, product_id, add_to_cart_order, reordered
    # products.csv:商品信息
        # 字段:product_id, product_name,aisle_id,department_id
    # orders.csv:用户的订单信息
        # 字段:order_id,user_id,eval_set,order_number,.....
    # aisles.csv:商品所属具体物品类别
        # 字段:aisle_id,aisle
    tab1 = pd.merge(aisles,products,on=["aisle_id","aisle_id"])
    tab2 = pd.merge(tab1,order_products__prior,on=["product_id","product_id"])
    tab3 = pd.merge(tab2,orders,on=["order_id","order_id"])

    # 3.找user_id和aisle之间的关系
    table = pd.crosstab(tab3["user_id"],tab3["aisle"])
    data = table[:1000]

    # 4.PCA降维
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)

    print("data_new:\n",data_new)

    return data_new


def k_means_Cluster(data):
    """
    使用K_Means算法对数据进行分类
    :param data: PCA降维后的数据
    :return:
    """
    estimator = k_means(n_clusters=3)
    estimator.fit(data)
    estimator.predict(estimator)
    return None


if __name__ == '__main__':
    data = data_processing()
    k_means_Cluster(data)