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

#绘制数据集
def plotDataSet():
    #加载数据集
    xArr, yArr = loadDataSet('ex0.txt')
    #计算数据的个数
    n = len(xArr)
    #样本点
    xcord = []; ycord = []
    for i in range(n):
        xcord.append(xArr[i][1]); ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    plotDataSet()
