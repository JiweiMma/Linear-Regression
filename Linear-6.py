from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

#加载数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = [];
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

#数据标准化，# inxMat - 标准化后的x数据集
def regularize(xMat, yMat):
    #数据拷贝
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    # 行与行操作，求均值
    yMean = np.mean(yMat, 0)
    #数据减去均值
    inyMat = yMat - yMean
    # 行与行操作，求均值
    inMeans = np.mean(inxMat, 0)
    # 行与行操作，求方差
    inVar = np.var(inxMat, 0)
    # 数据减去均值除以方差实现标准化
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat

#计算平方误差
def rssError(yArr, yHatArr):
       # yArr - 预测值
       # yHatArr - 真实值
    return ((yArr - yHatArr) ** 2).sum()

#前向逐步线性回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
        #xArr - x输入数据 # yArr - y预测数据 #eps - 每次迭代需要调整的步长
        #numIt - 迭代次数      #returnMat - numIt次迭代的回归系数矩阵
    # 数据集
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    # 数据标准化
    xMat, yMat = regularize(xMat, yMat)
    m, n = np.shape(xMat)
    # 初始化numIt次迭代的回归系数矩阵
    returnMat = np.zeros((numIt, n))
    # 初始化回归系数矩阵
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    # 迭代numIt次
    for i in range(numIt):
        # print(ws.T)          #可以打印当前回归系数矩阵
        # 正无穷
        lowestError = float('inf');
        # 遍历每个特征的回归系数
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                # 微调回归系数
                wsTest[j] += eps * sign
                # 计算预测值
                yTest = xMat * wsTest
                # 计算平方误差
                rssE = rssError(yMat.A, yTest.A)
                # 如果误差更小，则更新当前的最佳回归系数
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        # 记录numIt次迭代的回归系数矩阵
        returnMat[i, :] = ws.T
    return returnMat

#绘制岭回归系数矩阵
def plotstageWiseMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotstageWiseMat()
