from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def scaleDataVector(X):
    scaler = StandardScaler().fit(X)

    return scaler.transform(X)

def scaleDataSet(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset[dataset.columns] = scaler.fit_transform(dataset[dataset.columns])
    return dataset


def readDataset(datasetID):
    if datasetID == 1:
        whlData = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/datapreparation/data/datasetv2.2.csv", sep=',')
        inputPref = "I"
        selectedCol = []
        i = 1
        while True:
            crrnCol = inputPref + str(i)
            if crrnCol in whlData.columns:
                selectedCol.append(crrnCol)
            else:
                break
            i += 1
        i = 1
        outputPref = "O"
        while True:
            crrnCol = outputPref + str(i)
            if crrnCol in whlData.columns:
                selectedCol.append(crrnCol)
            else:
                break
            i += 1
        return whlData[selectedCol]

    else:
        pass

def datasetToXY(dataset):
    inputPref = "I"
    inpCol = []
    i = 1
    while True:
        crrnCol = inputPref + str(i)
        if crrnCol in dataset.columns:
            inpCol.append(crrnCol)
        else:
            break
        i += 1
    i = 1
    outputPref = "O"
    outCol = []
    while True:
        crrnCol = outputPref + str(i)
        if crrnCol in dataset.columns:
            outCol.append(crrnCol)
        else:
            break
        i += 1
    return dataset[inpCol].values, dataset[outCol].values

def getData():
    dataset = readDataset(1)
    dataset = scaleDataSet(dataset)
    return  datasetToXY(dataset)
