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

#计算回归系数w
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

#绘制回归曲线和数据
def plotRegression():
    #加载数据集
    xArr, yArr = loadDataSet('ex0.txt')
    #计算回归系数
    ws = standRegres(xArr, yArr)
    #创建矩阵
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    #拷贝矩阵
    xCopy = xMat.copy()
    #排序
    xCopy.sort(0)
    # 计算对应的y值
    yHat = xCopy * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制回归曲线
    ax.plot(xCopy[:, 1], yHat, c = 'red')
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5)                #绘制样本点
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    plotRegression()
