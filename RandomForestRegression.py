'''
随机森林：回归问题
'''
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''计算总方差R2'''
def calcR2(dataSet):
    return np.var(dataSet) * len(dataSet) if len(dataSet) > 0 else 0
'''计算叶子节点'''
def calcLeaf(dataSet):
    return np.mean(dataSet[:, -1]) if len(dataSet) > 0 else 0
'''数据的随机抽样：有放回式抽样'''
def baggingDataSet(dataSet):
    n, m = dataSet.shape
    rows = [random.randint(0, n - 1) for _ in range(n)]  # 行采样：有放回抽样
    trainData = dataSet.iloc[rows][:]  # 根据随机特征有放回选取n个样本
    return trainData.values
'''特征的随机抽样：无放回式抽样'''
def baggingFeature(allFeature):
    size = int(np.sqrt(len(allFeature) - 1))  # 抽取的特征维度为m=sqrt(M-1)
    features = random.sample(list(allFeature[:-1]), size)  # 列采样：无放回抽样（特征的随机选取）
    features.append(allFeature[-1])  # 末尾添加标签特征
    return features
'''CART算法：将数据集二叉树划分，value：分割点'''
def splitData(dataSet, feature, value):
    leftData, rightData = [], []
    for example in dataSet:
        leftData.append(example) if example[feature] <= value else rightData.append(example)
    return leftData, rightData
'''寻找节点的最佳分裂方式'''
def chooseBestSplit(dataSet, features):
    bestR2, bestFeature, bestSplitValue = float('inf'), -1, None
    # 尝试不同feature来分裂节点
    for feature in features[:-1]:
        featureList = [example[feature] for example in dataSet]  # 获取所有特征值
        # 按照特征排序，由相邻离散值的中点为候选点（准备进行分裂）
        sortfeatureList = sorted(list(set(featureList)))
        splitList = []
        if len(sortfeatureList) == 1:  # 如果值相同，不存在候选划分点
            splitList.append(sortfeatureList[0])
        else:
            for j in range(len(sortfeatureList) - 1):
                splitList.append((sortfeatureList[j] + sortfeatureList[j + 1]) / 2)
        # 遍历所有候分割值
        for splitValue in splitList:
            subDataSet0, subDataSet1 = splitData(dataSet=dataSet, feature=feature, value=splitValue)  # 按照feature分裂
            R2 = calcR2(dataSet=subDataSet0) + calcR2(dataSet=subDataSet1)  # 计算R2总方差
            # 用R2总方差评分选择出最佳feature和分割点
            if R2 < bestR2:
                bestR2 = R2
                bestFeature = feature
                bestSplitValue = splitValue
    return bestFeature, bestSplitValue
'''递归建立决策树'''
def createRegressionTree(dataSet, allFeatures):
    randomFeatures = baggingFeature(allFeature=allFeatures)  # 无放回随机选择m个feature
    classList = [example[-1] for example in dataSet]
    # 停止分裂条件：当前节点数据集的类别都一样
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 选出使得分裂效果最佳的特征和分割点值
    bestFeature, bestSplitValue = chooseBestSplit(dataSet=dataSet, features=randomFeatures)
    # 用bestFeature和bestSplitValue分裂树节点
    leftData, rightData = splitData(dataSet, feature=bestFeature, value=bestSplitValue)
    # 左右子树有一个为空，则返回该节点下样本均值
    if len(leftData) == 0 or len(rightData) == 0:
        return calcLeaf(dataSet=np.array(leftData)) + calcLeaf(dataSet=np.array(rightData))
    else:
        myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
        myTree[bestFeature]['<' + str(bestSplitValue)] = createRegressionTree(leftData, allFeatures[:])
        myTree[bestFeature]['>' + str(bestSplitValue)] = createRegressionTree(rightData, allFeatures[:])
    return myTree
'''单颗决策树对测试样本进行分类'''
def regressionTreeClassify(regressionTree, featureLabel, testVec):
    firstFeature = list(regressionTree.keys())[0]
    secondFeatDict = regressionTree[firstFeature]
    splitValue = float(list(secondFeatDict.keys())[0][1:])
    featureIndex = featureLabel.index(firstFeature)
    if testVec[featureIndex] <= splitValue:
        valueOfFeat = secondFeatDict['<' + str(splitValue)]
    else:
        valueOfFeat = secondFeatDict['>' + str(splitValue)]
    if isinstance(valueOfFeat, dict):
        pred_label = regressionTreeClassify(valueOfFeat, featureLabel, testVec)
    else:
        pred_label = valueOfFeat
    return pred_label
'''随机森林交叉验证正确率'''
def accRandomForest(dataSet, labels, randomForest):
    predicts = []
    for example in dataSet.values:
        # k颗决策树对测试样本预测分类
        labelPred = []
        for tree in randomForest:
            label = regressionTreeClassify(tree, labels[:-1], testVec=example[:-1])
            labelPred.append(label)
        # k颗决策树预测值通过求平均值决定森林预测值
        predict = np.mean(labelPred)
        predicts.append(predict)
    return predicts
'''执行随机森林分类'''
def setupRandomForest(dataSet, K=11):
    randomForest = []  # 随机森林
    for i in range(K):  # K颗决策树
        baggingData = baggingDataSet(dataSet=dataSet)  # 有放回式随机抽取N个样本子集
        regressionTree = createRegressionTree(dataSet=baggingData, allFeatures=list(range(len(dataSet.columns))))  # 构造决策树
        randomForest.append(regressionTree)  # 第i颗决策树加入森林
    return randomForest
'''画图'''
def draw(y, yHat):
    plt.plot(np.arange(0, len(y), 1), y, color='b', label='true value')
    plt.plot(np.arange(0, len(yHat), 1), yHat, color='r', label='predict value')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    df = pd.read_csv('./data/rf_Regression.csv')
    df = df.iloc[np.random.permutation(len(df))]  # 打乱df
    splitRate = int(len(df) * 0.8)  # 交叉验证比率
    trainDF, testDF = df[0: splitRate], df[splitRate:]  # 训练集和测试集
    randomForest = setupRandomForest(dataSet=trainDF, K=10)  # 随机森林模型
    labels = list(range(len(df.columns)))  # 所有特征列集合
    yHat = accRandomForest(dataSet=testDF, labels=labels, randomForest=randomForest)  # 交叉检测模型
    draw(y=testDF.iloc[:, -1].values, yHat=yHat)

    '''sklearn演示对比'''
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # from sklearn.ensemble import RandomForestRegressor
    # regressor = RandomForestRegressor(n_estimators=10, max_features='sqrt', random_state=123)
    # regressor.fit(X_train, y_train)
    # yHat = regressor.predict(X_test)
    # draw(y=y_test, yHat=yHat)
