from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

#加载数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

#岭回归
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    # lam - 缩减系数
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

#岭回归测试
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    # 行与行操作，求均值
    yMean = np.mean(yMat, axis = 0)
    # 数据减去均值
    yMat = yMat - yMean
    # 行与行操作，求均值
    xMeans = np.mean(xMat, axis = 0)
    # 行与行操作，求方差
    xVar = np.var(xMat, axis = 0)
    # 数据减去均值除以方差实现标准化
    xMat = (xMat - xMeans) / xVar
    # 30个不同的lambda测试
    numTestPts = 30
    # 初始回归系数矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # 改变lambda计算回归系数
    for i in range(numTestPts):
        # lambda以e的指数变化
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        # 计算回归系数矩阵
        wMat[i, :] = ws.T
    return wMat

#绘制岭回归系数矩阵
def plotwMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()
if __name__ == '__main__':
    plotwMat()
