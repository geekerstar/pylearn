import jieba
import joblib
import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import fetch_california_housing
import numpy as np

def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris["DESCR"])
    print("查看特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)

    return iris["DESCR"]


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=True)

    # 2、调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(), type(data_new))
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def count_demo():
    """
    文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is", "too"])

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def count_chinese_demo():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def cut_word(text):
    """
    进行中文分词："我爱北京天安门" --> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种", "所以"])

    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])

    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = MinMaxScaler(feature_range=[2, 3])

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = StandardScaler()

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = VarianceThreshold(threshold=10)

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)

    # 计算某两个变量之间的相关系数
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数：\n", r1)
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("revenue与total_expense之间的相关性：\n", r2)

    return None


def pca_demo():
    """
    PCA降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1、实例化一个转换器类
    transfer = PCA(n_components=0.95)

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1）获取数据
    iris = load_iris()

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None


def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    :return:
    """
    # 1）获取数据
    iris = load_iris()

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）KNN算法预估器
    estimator = KNeighborsClassifier()

    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 5）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 最佳参数：best_params_
    print("最佳参数：\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果：\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果:\n", estimator.cv_results_)

    return None


def nb_news():
    """
    用朴素贝叶斯算法对新闻进行分类
    :return:
    """
    # 1）获取数据
    news = fetch_20newsgroups(subset="all")

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3）特征工程：文本特征抽取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None


def decision_iris():
    """
    用决策树对鸢尾花进行分类
    :return:
    """
    # 1）获取数据集
    iris = load_iris()

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)

    return None

def linear1():
    """
    正规方程的优化方法对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    boston = fetch_california_housing()

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5）得出模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)

    return None


def linear2():
    """
    梯度下降的优化方法对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    boston = fetch_california_housing()
    print("特征数量：\n", boston.data.shape)

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000, penalty="l1")
    estimator.fit(x_train, y_train)

    # 5）得出模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为：\n", error)

    return None


def linear3():
    """
    岭回归对波士顿房价进行预测
    :return:
    """
    # 1）获取数据
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    boston = fetch_california_housing()
    print("特征数量：\n", boston.data.shape)

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3）标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4）预估器
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 保存模型
    joblib.dump(estimator, "my_ridge.pkl")
    # 加载模型
    estimator = joblib.load("my_ridge.pkl")

    # 5）得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)

    # 6）模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差为：\n", error)

    return None

if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()
    # 代码2：字典特征抽取
    # dict_demo()
    # 代码3：文本特征抽取：CountVecotrizer
    # count_demo()
    # 代码4：中文文本特征抽取：CountVecotrizer
    # count_chinese_demo()
    # 代码5：中文文本特征抽取，自动分词
    # count_chinese_demo2()
    # 代码6：中文分词
    # print(cut_word("我爱北京天安门"))
    # 代码7：用TF-IDF的方法进行文本特征抽取
    # tfidf_demo()
    # 代码8：归一化
    # minmax_demo()
    # 代码9：标准化
    # stand_demo()
    # 代码10：低方差特征过滤
    # variance_demo()
    # 代码11：PCA降维
    # pca_demo()
    # 代码1： 用KNN算法对鸢尾花进行分类
    # knn_iris()
    # 代码2：用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    # knn_iris_gscv()
    # 代码3：用朴素贝叶斯算法对新闻进行分类
    # nb_news()
    # 代码4：用决策树对鸢尾花进行分类
    # decision_iris()
    # 代码1：正规方程的优化方法对波士顿房价进行预测
    # linear1()
    # 代码2：梯度下降的优化方法对波士顿房价进行预测
    # linear2()
    # 代码3：岭回归对波士顿房价进行预测
    linear3()