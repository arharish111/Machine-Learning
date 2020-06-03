import numpy as np
import os
import random
import sys
from sklearn.feature_extraction.text import CountVectorizer
random.seed(3)
targetEncoder = {
    'alt.atheism':1,
    'comp.graphics':2,
    'comp.os.ms-windows.misc':3,
    'comp.sys.ibm.pc.hardware':4, 
    'comp.sys.mac.hardware':5, 
    'comp.windows.x':6, 
    'misc.forsale':7, 
    'rec.autos':8, 
    'rec.motorcycles':9, 
    'rec.sport.baseball':10, 
    'rec.sport.hockey':11, 
    'sci.crypt':12, 
    'sci.electronics':13, 
    'sci.med':14, 
    'sci.space':15, 
    'soc.religion.christian':16, 
    'talk.politics.guns':17, 
    'talk.politics.mideast':18, 
    'talk.politics.misc':19, 
    'talk.religion.misc':20
}

def main(basePath):
    trainTarget = []
    testTarget = []
    corpusTrainData = []
    corpusTestData = []
    for directory in targetEncoder.keys():
        filePath = os.path.join(basePath,directory)
        files = os.listdir(filePath)
        length = len(files)
        trainFiles = length/2
        count = 0
        trainData = []
        testData = []
        random.shuffle(files)
        for file in files:
            count += 1
            with (open(os.path.join(filePath,file),encoding='utf8',errors='ignore')) as f:
                readLines = f.readlines()
            data = ''
            for line in readLines:
                if line[:4] != 'Xref' and line[:4] != 'Path' and line[:10] != 'Newsgroups' and line[:10] != 'Message-ID'\
                and line[:4] !='Date' and line[:7] != 'Expires' and line[:11] != 'Followup-To' and line[:5] != 'Lines':
                    data += line
            if count <= trainFiles:
                trainData.append(data)
                trainTarget.append(targetEncoder[directory])
            else:
                testData.append(data)
                testTarget.append(targetEncoder[directory])
        corpusTrainData.append(trainData)
        corpusTestData.append(testData)

    numberOfSamples = 0
    for tp in corpusTrainData:
        numberOfSamples += len(tp)

    count_vect = CountVectorizer(stop_words='english',token_pattern='[a-z]{4,20}')
    corpusLikelyhood = []
    corpusPrior = []
    corpusSumOfTokensPlusV = []
    for topic in corpusTrainData:
        numOfDocs = len(topic)
        corpusPrior.append(numOfDocs/numberOfSamples)
        X_train_counts = count_vect.fit_transform(topic)
        likelyhoodDict = {}
        tokens = count_vect.get_feature_names()
        documentFrequency = X_train_counts.toarray()
        tokenFrequency = np.sum(documentFrequency,axis=0)
        totalSumOfTokensPlusV = np.sum(documentFrequency)
        corpusSumOfTokensPlusV.append(totalSumOfTokensPlusV)
        for tk in range(len(tokens)):
            likelyhoodDict[tokens[tk]] = tokenFrequency[tk]
        corpusLikelyhood.append(likelyhoodDict)

    m = 0.0001
    corpusClass = []
    totaTestCases = 0
    for testTopic in range(len(corpusTestData)):
        predictedClass = []
        for testDoc in corpusTestData[testTopic]:
            totaTestCases += 1
            maxPosterior = 0
            pClass = -1
            X_test_counts = count_vect.fit_transform([testDoc])
            tokenFrequency = X_test_counts.toarray()
            tokens = count_vect.get_feature_names()
            for topic in range(len(corpusLikelyhood)):
                d = int(m*corpusSumOfTokensPlusV[topic])
                posterior1 = []
                posterior2 = []
                tkf = []
                lDict = corpusLikelyhood[topic]
                for tk in range(len(tokens)):
                    if tokens[tk] in lDict:
                        posterior1.append(lDict[tokens[tk]])
                        tkf.append(tokenFrequency[0][tk])
                    else:
                        posterior2.append(d)
                denominator = corpusSumOfTokensPlusV[topic] + (len(posterior2)*d)
                tkfArray = np.array(tkf)
                p1Array = np.array(posterior1)/denominator
                p1Array = p1Array + 1
                p1Array = p1Array**tkfArray
                if len(posterior2)>0:
                    p2Array = np.array(posterior2)/denominator
                    posterior = np.prod(np.concatenate((p1Array,p2Array))) * corpusPrior[topic]
                else:
                    posterior = np.prod(p1Array) * corpusPrior[topic]
                if posterior>maxPosterior:
                    maxPosterior = posterior
                    pClass = topic + 1
            predictedClass.append(pClass)
        corpusClass.append(predictedClass)

    count = 0
    tsum = 0
    print('Individual Accuracy:\n')
    for p in corpusClass:
        count += 1
        pp = np.asarray(p)
        t = np.count_nonzero(pp == count)
        print(f'{count}:{round((t/pp.size)*100,2)}%')
        tsum += t
    print('\n')
    print(f'Accuracy:{round((tsum/totaTestCases)*100,2)}%')

if __name__ == "__main__":
    main(sys.argv[1])