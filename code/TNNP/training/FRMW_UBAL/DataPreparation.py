from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import random
from sklearn.externals import joblib
import numpy as np

def scaleDataVector(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def scaleDataSet(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    print(dataset.columns)
    dataset[dataset.columns] = scaler.fit_transform(dataset[dataset.columns])
    joblib.dump(scaler, "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc4.pkl")
    dataset.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/last_saved_scaled_3.csv")

    return dataset


def readDataset(path):
    whlData = pd.read_csv(path, sep=',')
    inputPref = "I"
    selectedCol = []
    i = 0
    while True:
        crrnCol = inputPref + str(i)
        if crrnCol in whlData.columns:
            selectedCol.append(crrnCol)
        else:
            if i is not 0:
                break

        i += 1
    i = 0
    outputPref = "O"
    while True:
        crrnCol = outputPref + str(i)
        if crrnCol in whlData.columns:
            selectedCol.append(crrnCol)
        else:
            if i is not 0:
                break
        i += 1
    return whlData[selectedCol]

def datasetToXY(dataset):
    inputPref = "I"
    inpCol = []
    i = 0
    while True:
        crrnCol = inputPref + str(i)
        if crrnCol in dataset.columns:
            inpCol.append(crrnCol)
        else:
            if i is not 0:
                break
        i += 1
    i = 0
    outputPref = "O"
    outCol = []
    while True:
        crrnCol = outputPref + str(i)
        if crrnCol in dataset.columns:
            outCol.append(crrnCol)
        else:
            if i is not 0:
                break
        i += 1
    return dataset[inpCol].values, dataset[outCol].values

def splitDataSetToTestAndTrain(X, Y, ratio=0.7):
    print(X.shape, Y.shape)
    size = X.shape[0]
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same count of samples")

    train_size = int(size*0.7)
    indexs = set(range(0, size))
    train_idx = set(random.sample(indexs, train_size))
    test_idx = indexs - train_idx

    return X[list(train_idx)], Y[list(train_idx)], X[list(test_idx)], Y[list(test_idx)]



def getData(path, scale=True):
    dataset = readDataset(path)
    if scale:
        dataset = scaleDataSet(dataset)
    return datasetToXY(dataset)

def mergePointsWithPredictions(version="ret"):
    if version != "ret":
        df_test = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/real_datasetv2.csv")
        df_real = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/new_flt.csv")
        df_points = pd.read_csv("/home/martin/testing/ret_pointsv2.csv")


        test_data = df_test[['I0','I1','I2','O0','O1','O2','O3','O4','O5','O6']]
        real_data = df_real[['I1','I2','I3','O1','O2','O3','O4','O5','O6','O7']]

        df_new = pd.DataFrame(columns=['h1', 'h2', 'h3', 'a1', 'a2', 'a3', 'f1', 'f2', 'f3', 'x1', 'x2', 'x3'])

        for i in range(test_data.shape[0]):
            for j in range(real_data.shape[0]):
                if np.array_equal(np.around(test_data.iloc[i], 4), np.around(real_data.iloc[j], 4)):
                    df_new.loc[i] = np.concatenate((df_points.iloc[i].values, df_real.iloc[j][['x1', 'x2', 'x3', 'y1', 'y2', 'y3']].values), axis=0).tolist()
                    print("{} - DONE!".format(i+1))
                    break

        df_new.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/PointsToCompared.csv")

    else:
        df_test = pd.read_csv(
            "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/real_datasetv2.csv")
        df_real = pd.read_csv(
            "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/new_flt.csv")
        df_points = pd.read_csv("/home/martin/data/testing/ret_pointsv2.csv")

        test_data = df_test[['I0', 'I1', 'I2', 'O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6']]
        real_data = df_real[['I1', 'I2', 'I3', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']]

        df_new = pd.DataFrame(columns=['a1', 'a2', 'a3', 'h1', 'h2', 'h3', 'f1', 'f2', 'f3'])

        for i in range(test_data.shape[0]):
            for j in range(real_data.shape[0]):
                if np.array_equal(np.around(test_data.iloc[i], 4), np.around(real_data.iloc[j], 4)):
                    df_new.loc[i] = np.concatenate(
                        (df_points.iloc[i].values, df_real.iloc[j][['FX', 'FY', 'FZ']].values),
                        axis=0).tolist()
                    print("{} - DONE!".format(i + 1))
                    break
        df_new.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/PointsToComparedv2.csv")

if __name__ == '__main__':
    mergePointsWithPredictions(version="ret")