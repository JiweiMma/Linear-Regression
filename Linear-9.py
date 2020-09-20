import numpy as np
from bs4 import BeautifulSoup
import random


#从页面读取数据，生成列表
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    i = 1
    # 根据HTML页面结构进行解析
    #以列表形式返回符合条件的节点
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)

#岭回归   lam缩减系数
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


#依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(retX, retY):
    # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)
    # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)
    # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)
    # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)
    # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)
    # 2009年的乐高10196,部件数目3263,原价249.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)

#数据标准化
def regularize(xMat, yMat):
    #数据拷贝
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    # 行与行操作，求均值
    yMean = np.mean(yMat, 0)
    # 数据减去均值
    inyMat = yMat - yMean
    # 行与行操作，求均值
    inMeans = np.mean(inxMat, 0)
    # 行与行操作，求方差
    inVar = np.var(inxMat, 0)
    # print(inxMat)
    print(inMeans)
    # print(inVar)
    # 数据减去均值除以方差实现标准化
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat


#计算平方误差
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


#计算回归系数w
def standRegres(xArr, yArr):
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

#交叉验证岭回归
def crossValidation(xArr, yArr, numVal=10):
    # 统计样本个数
    m = len(yArr)
    # 生成索引值列表
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    # 交叉验证numVal次
    for i in range(numVal):
        trainX = [];
        # 训练集
        trainY = []
        testX = [];
        # 测试集
        testY = []
        # 打乱次序
        random.shuffle(indexList)
        # 划分数据集:90%训练集，10%测试集
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 获得30个不同lambda下的岭回归系数
        wMat = ridgeTest(trainX, trainY)
        # 遍历所有的岭回归系数
        for k in range(30):
            matTestX = np.mat(testX);
            # 测试集
            matTrainX = np.mat(trainX)
            # 测试集均值
            meanTrain = np.mean(matTrainX, 0)
            # 测试集方差
            varTrain = np.var(matTrainX, 0)
            # 测试集标准化
            matTestX = (matTestX - meanTrain) / varTrain
            # 根据ws预测y值
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            # 统计误差
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
    # 计算每次交叉验证的平均误差
    meanErrors = np.mean(errorMat, 0)
    # 找到最小误差
    minMean = float(min(meanErrors))
    # 找到最佳回归系数
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0);
    varX = np.var(xMat, 0)
    # 数据经过标准化，因此需要还原
    unReg = bestWeights / varX
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (
    (-1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat)), unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))

#岭回归测试
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T
    # 行与行操作，求均值
    yMean = np.mean(yMat, axis=0)
    # 数据减去均值
    yMat = yMat - yMean
    # 行与行操作，求均值
    xMeans = np.mean(xMat, axis=0)
    # 行与行操作，求方差
    xVar = np.var(xMat, axis=0)
    # 数据减去均值除以方差实现标准化
    xMat = (xMat - xMeans) / xVar
    # 30个不同的lambda测试
    numTestPts = 30
    # 初始回归系数矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # 改变lambda计算回归系数
    for i in range(numTestPts):
        # lambda以e的指数变化，最初是一个非常小的数，
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        # 计算回归系数矩阵
        wMat[i, :] = ws.T  # 计算回归系数矩阵
    return wMat


if __name__ == '__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    crossValidation(lgX, lgY)
