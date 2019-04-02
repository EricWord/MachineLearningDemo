'''
KNN k-近邻算法
使用该算法对鸢尾花进行分类预测
sklearn
    参数 n-neighbor 对应k值
    使用步骤：
    1、获取数据
    2、数据集划分
    3、特征工程-->标准化
    4、KNN算法预估值
    5、模型评估


'''
# 导入训练集

from sklearn.datasets import load_iris

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV


def knn_iris():
    # 1、获取数据
    iris=load_iris()
    # 2、数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=6)
    # 3、特征工程-->标准化
    transfer=StandardScaler()
    # 训练集标准化
    x_train=transfer.fit_transform(x_train)
    # 训练集和测试集进行相同的处理
    x_test=transfer.transform(x_test)
    # 4、KNN算法预估值
    estimator=KNeighborsClassifier(n_neighbors=22)
    estimator.fit(x_train,y_train)
    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("比对真实值和预测值：\n",y_test==y_predict)

    # 方法2：计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为：\n",score)

    return  None
def knn_iris_qscv():
    '''
    用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    :return:
    '''

    # 1、获取数据
    iris=load_iris()
    # 2、数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=6)
    # 3、特征工程-->标准化
    transfer=StandardScaler()
    # 训练集标准化
    x_train=transfer.fit_transform(x_train)
    # 训练集和测试集进行相同的处理
    x_test=transfer.transform(x_test)
    # 4、KNN算法预估值
    estimator=KNeighborsClassifier()
    # 加入网格搜搜和交叉验证
    # 参数准备
    param_dict={"n_neighbors":[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_train)
    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("比对真实值和预测值：\n",y_test==y_predict)

    # 方法2：计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为：\n",score)
    # 最佳参数
    print("最佳参数:\n",estimator.best_params_)
    # 最佳结果
    print("最佳结果：\n",estimator.best_score_)
    # 最佳估计器
    print("最佳估计器：\n",estimator.best_estimator_)
    # 交叉验证结果
    print("交叉验证结果：\n",estimator.cv_results_)

    return  None

if __name__=='__main__':
    # knn_iris()
    knn_iris_qscv()

