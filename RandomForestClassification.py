'''
随机森林：分类问题
'''
import pandas as pd
import numpy as np
import random
import copy
'''计算基尼系数'''
def calcGini(dataSet):
    labelCounts = {}
    for example in dataSet:
        currentLabel = example[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    Gini = 1.0
    for key in labelCounts:
        prob = labelCounts[key] / len(dataSet)
        Gini -= prob * prob
    return Gini
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
'''CART算法：将数据集进行二叉树分裂，value：分割点值'''
def splitData(dataSet, feature, value):
    leftData, rightData = [], []
    for example in dataSet:
        if example[feature] <= value:
            leftData.append(example)
        else:
            rightData.append(example)
    return leftData, rightData
'''寻找节点的最佳分裂方式'''
def chooseBestSplit(dataSet, features):
    bestGini, bestFeature, bestSplitValue = 1, -1, None
    # 尝试不同feature来分裂节点
    for feature in features[:-1]:
        featureList = [example[feature] for example in dataSet]
        # 按照特征值排序，由相邻离散值的中点为分割点（准备进行分裂）
        sortfeatureList = sorted(list(set(featureList)))
        splitList = []
        for j in range(len(sortfeatureList) - 1):
            splitList.append((sortfeatureList[j] + sortfeatureList[j + 1]) / 2)
        # 遍历所有分割点
        for splitValue in splitList:
            newGini = 0
            subDataSet0, subDataSet1 = splitData(dataSet=dataSet, feature=feature, value=splitValue)
            newGini += len(subDataSet0) / len(dataSet) * calcGini(subDataSet0)
            newGini += len(subDataSet1) / len(dataSet) * calcGini(subDataSet1)
            # 用Gini系数评分选择出最佳feature和分割点
            if newGini < bestGini:
                bestGini = newGini
                bestFeature = feature
                bestSplitValue = splitValue
    return bestFeature, bestSplitValue
'''递归建立决策树'''
def createTree(dataSet, allFeatures):
    randomFeatures = baggingFeature(allFeature=allFeatures)  # 无放回随机选择m个feature
    classList = [example[-1] for example in dataSet]
    # 停止分裂条件：当前节点数据集的类别都一样
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 选出使得分裂效果最佳的特征和分割点值
    bestFeature, bestSplitValue = chooseBestSplit(dataSet=dataSet, features=randomFeatures)
    # 用bestFeature和bestSplitValue分裂树节点
    leftData, rightData = splitData(dataSet=dataSet, feature=bestFeature, value=bestSplitValue)
    # 左右两颗子树，左边小于等于最佳划分点，右边大于最佳划分点
    myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
    myTree[bestFeature]['<' + str(bestSplitValue)] = createTree(leftData, allFeatures[:])
    myTree[bestFeature]['>' + str(bestSplitValue)] = createTree(rightData, allFeatures[:])
    return myTree
'''单颗决策树对测试样本进行分类'''
def treeClassify(decisionTree, featureLabel, testVec):
    firstFeature = list(decisionTree.keys())[0]
    secondFeatDict = decisionTree[firstFeature]
    splitValue = float(list(secondFeatDict.keys())[0][1:])
    featureIndex = featureLabel.index(firstFeature)
    if testVec[featureIndex] <= splitValue:
        valueOfFeat = secondFeatDict['<' + str(splitValue)]
    else:
        valueOfFeat = secondFeatDict['>' + str(splitValue)]
    if isinstance(valueOfFeat, dict):
        pred_label = treeClassify(valueOfFeat, featureLabel, testVec)
    else:
        pred_label = valueOfFeat
    return pred_label
'''随机森林交叉验证正确率'''
def accRandomForest(dataSet, labels, randomForest):
    accuracy = 0
    for example in dataSet.values:
        # k颗决策树对测试样本预测分类
        labelPred = []
        for tree in randomForest:
            label = treeClassify(tree, labels[:-1], testVec=example[:-1])
            labelPred.append(label)
        # k颗决策树预测类别通过投票选择森林预测的最终类别
        labelDict = {}
        for label in labelPred:
            labelDict[label] = labelDict.get(label, 0) + 1
        sortClass = sorted(labelDict.items(), key=lambda item: item[1], reverse=True)
        # 计算模型预测的正确率
        if sortClass[0][0] == example[-1:][0]:
            accuracy += 1
    print('测试样本：{}个，预测正确：{}个，预测错误：{}个，随机森林分类准确率为：{:.2%}'.
          format(len(dataSet), accuracy, len(dataSet) - accuracy, accuracy / len(dataSet)))
'''执行随机森林分类'''
def setupRandomForest(dataSet, K=100):
    randomForest = []  # 随机森林
    for i in range(K):  # K颗决策树
        baggingData = baggingDataSet(dataSet=dataSet)  # 有放回式随机抽取N个样本子集
        decisionTree = createTree(dataSet=baggingData, allFeatures=dataSet.columns.values)  # 构造决策树
        randomForest.append(decisionTree)  # 第i颗决策树加入森林
    return randomForest
if __name__ == '__main__':
    df = pd.read_csv('./data/rf_Classification.csv', header=None)
    labels = df.columns.values.tolist()  # 所有特征列集合
    df = df.iloc[np.random.permutation(len(df))]  # 打乱df
    splitRate = int(len(df) * 0.8)  # 交叉验证比率
    trainDF, testDF = df[0: splitRate], df[splitRate:]  # 训练集和测试集
    randomForest = setupRandomForest(dataSet=trainDF, K=11)  # 随机森林模型
    accRandomForest(dataSet=testDF, labels=labels, randomForest=randomForest)  # 交叉检测模型
