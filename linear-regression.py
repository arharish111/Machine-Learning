import numpy as np
import pandas as pd
from sklearn import preprocessing

class LinearRegression(object):
    def __init__(self,filePath):
        self.data = pd.read_csv(filePath,header=None)
        self.dataArray = np.array([])
        self.W = np.array([])
        self.labelEncoder = preprocessing.LabelEncoder()
    def preprocess(self,seed=None):
        self.data[4] = self.labelEncoder.fit_transform(self.data[4])
        self.dataArray = self.data.to_numpy()
        if seed != None:
            np.random.seed(seed)
        np.random.shuffle(self.dataArray)
    def getWeights(self):
        return self.W
    def setWeights(self,W):
        self.W = W
    def getPreprocessedData(self):
        return self.dataArray
    def predict(self,A,W):
        predictedOutput = np.dot(A,W)
        po = np.where(predictedOutput <= 0.45,0,predictedOutput)
        po = np.where((po>0.45)&(po<=1.45),1,po)
        po = np.where(po>1.45,2,po)
        return po
    def calculateBeta(self,A,Y):
        AtA_inverse = np.linalg.inv(np.dot(A.T,A))
        AtY = np.dot(A.T,Y)
        W = np.dot(AtA_inverse,AtY)
        return W
    def train(self,fold=3):
        partition = np.array_split(self.dataArray,fold)
        leastError = 100.0
        for k in range(fold):
            xTrain = np.array([])
            xTest = np.array([])
            for f in range(fold):
                if k==f:
                    xTest = partition[f]
                else:
                    if xTrain.size==0:
                        xTrain = partition[f]
                    else:
                        xTrain = np.concatenate((xTrain,partition[f]))
            aTrain = xTrain[:,:4]
            aTest = xTest[:,:4]
            yTrain = xTrain[:,4]
            yTest = xTest[:,4]
            yTrain = np.expand_dims(yTrain,axis=1)
            yTest = np.expand_dims(yTest,axis=1)
            W = self.calculateBeta(aTrain,yTrain)
            po = self.predict(aTest,W)
            e = self.calculatePredictionError(po,yTest)
            if e<leastError:
                leastError = e
                self.W = W
    def calculatePredictionError(self,predicted,target):
        res = np.equal(predicted,target)
        return 100 * ((np.size(res) - np.count_nonzero(res))/np.size(res))
    def writeOutputToFile(self,i,o):
        out = pd.DataFrame(i)
        out[4] = out[4].astype(int)
        out[5] = o.astype(int)
        out[4] = self.labelEncoder.inverse_transform(out[4])
        out[5] = self.labelEncoder.inverse_transform(out[5])
        out.columns = ['sepal-length','sepal-width','petal-length','petal-width','target','predicted-output']
        out.to_csv('results.txt',index=False,sep=" ")

def calculateIndividualAccuracy(p,t):
    zeroCount = 0
    oneCount = 0
    twoCount = 0
    for i in range(p.shape[0]):
        if t[i][0]==0 and t[i][0]==p[i][0]:
            zeroCount += 1
        elif t[i][0]==1 and t[i][0]==p[i][0]:
            oneCount += 1
        elif t[i][0]==2 and t[i][0]==p[i][0]:
            twoCount += 1
    elements, counts = np.unique(t, return_counts=True)
    c = dict(zip(elements,counts))
    c[0] = 100 * (zeroCount/c[0])
    c[1] = 100 * (oneCount/c[1])   
    c[2] = 100 * (twoCount/c[2])
    return c  

if __name__ == "__main__":
    model = LinearRegression("iris.data")
    model.preprocess(seed=2)
    data = model.getPreprocessedData()
    model.train(fold=5)
    w = model.getWeights()
    print("Calculated Beta:",w)
    output = model.predict(data[:,:4],w)
    t = np.expand_dims(data[:,4],axis=1)
    percentageError = model.calculatePredictionError(output,t)
    accuracy = 100.0 - percentageError
    print("Accuracy:",accuracy)
    model.writeOutputToFile(data,output)
    print("Accuracy per class:",calculateIndividualAccuracy(output,t))