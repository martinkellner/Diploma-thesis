from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import random
from sklearn.externals import joblib
import numpy as np
import cv2
#from sklearn.preprocessing import scale
from training.FRMW_UBAL.populationCoder import *
from training.FRMW_UBAL.UbalNet import *
from training.FRMW_UBAL.populationCoder import *

def scaleDataVector(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def scaleDataSet(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset[dataset.columns] = scaler.fit_transform(dataset[dataset.columns])
    joblib.dump(scaler, "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc5.pkl")
    dataset.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/last_saved_scaled_5.csv")

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
    size = X.shape[0]
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same count of samples")

    train_size = int(size*ratio)
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
    if version == "1-1":
        df_test = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/_real.csv_u")
        df_real = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/datasetv2.3.csv")
        df_points = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/points.csv")


        test_data = df_test[['I0','I1','I2','O0','O1','O2','O3','O4','O5','O6']]
        real_data = df_real[['I1','I2','I3','O1','O2','O3','O4','O5','O6','O7']]

        df_new = pd.DataFrame(columns=['h1', 'h2', 'h3', 'a1', 'a2', 'a3', 'f1', 'f2', 'f3', 'x1', 'x2', 'x3'])

        for i in range(test_data.shape[0]):
            found = False
            for j in range(real_data.shape[0]):
                if np.array_equal(np.around(test_data.iloc[i], 4), np.around(real_data.iloc[j], 4)):
                    df_new.loc[i] = np.concatenate((df_points.iloc[i].values, df_real.iloc[j][['x1', 'x2', 'x3', 'y1', 'y2', 'y3']].values), axis=0).tolist()
                    print("{} - DONE!".format(i+1))
                    found = True
                    break
            if not found:
                print(("{}. sample not found".format(i)))

        df_new.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/PointsToCompared.csv")

    elif version == "ret":
        df_test = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/real_datasetv2.csv")
        df_real = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/new_flt.csv")
        df_points = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/21v_ret_point.csv")

        test_data = df_test[['I0', 'I1', 'I2', 'O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6']]
        real_data = df_real[['I1', 'I2', 'I3', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']]

        df_new = pd.DataFrame(columns=['a1', 'a2', 'a3', 'h1', 'h2', 'h3', 'f1', 'f2', 'f3', 'x1', 'x2', 'x3'])

        for i in range(test_data.shape[0]):
            found = False
            for j in range(real_data.shape[0]):
                if np.array_equal(np.around(test_data.iloc[i], 2), np.around(real_data.iloc[j], 2)):
                    df_new.loc[i] = np.concatenate(
                        (df_points.iloc[i].values, df_real.iloc[j][['FX', 'FY', 'FZ', 'E1', 'E2', 'E3']].values),
                        axis=0).tolist()
                    print("{} - DONE!".format(i + 1))
                    found = True
                    break
            if not found:
                print("Not found!")

        df_new.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/PointsToComparedv2.csv")
    elif version == "ret2":
        df_test = np.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/saved_dataset/4_xtes.npy")
        df_real = pd.read_csv(
            "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/new_flt.csv")

        df_points = pd.read_csv("/home/martin/data/testingV2/ret_point.csv")
        df_new = pd.DataFrame(columns=['a1', 'a2', 'a3', 'h1', 'h2', 'h3', 'f1', 'f2', 'f3'])

        real_data = df_real[['I1', 'I2', 'I3', 'I4', 'I5', 'I7', 'I8']]
        inpNeurons = [10, 12, 8, 32, 28, 32, 28]
        inpWidths = [4, 5, 4, 8, 6, 8, 6]

        for i in range(df_test.shape[0]):
            found = False
            for j in range(real_data.shape[0]):
                testXs = np.asarray(decodeInput(df_test[i,:], inpNeurons, inpWidths))
                print(np.around(testXs, 0), np.around(real_data.iloc[j].values, 0))

                if np.array_equal(np.around(testXs[:2], 1), np.around(real_data.iloc[j].values[:2], 1)) and np.array_equal(np.around(testXs[2:], 0), np.around(real_data.iloc[j].values[2:], 0)):
                    df_new.loc[i] = np.concatenate(
                        (df_points.iloc[i].values, df_real.iloc[j][['FX', 'FY', 'FZ']].values),
                        axis=0).tolist()
                    print("{} - DONE!".format(i + 1))
                    found = True

            if not found:
                print("Not found!")
        df_new.to_csv(
            "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/PointsToCompareV3.csv")

def rescaleImages(pathtodataset):
    dataset = pd.read_csv(pathtodataset)
    for path in dataset['I4']:
        filename = path
        W = 1000.
        oriimg = cv2.imread(filename)
        height, width, depth = oriimg.shape
        imgScale = 0.2
        newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
        newimg = cv2.resize(oriimg, (int(newX), int(newY)))
        cv2.imwrite("/home/martin/data/v2/filtered/scaled/{}".format(filename.split('/')[-1]), newimg)

    for path in dataset['I5']:
        filename = path
        W = 1000.
        oriimg = cv2.imread(filename)
        height, width, depth = oriimg.shape
        imgScale = 0.2
        newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
        newimg = cv2.resize(oriimg, (int(newX), int(newY)))

        cv2.imwrite("/home/martin/data/v2/filtered/scaled/{}".format(filename.split('/')[-1]), newimg)


    # Reshaping images to the shape (64*48,1) and scaling to the range <0,1>
    for path in dataset['I4']:
        image = cv2.imread("/home/martin/data/v2/filtered/scaled/{}".format(path.split('/')[-1]))
        image = image[:, :, 1]
        image = image.reshape((image.shape[0]*image.shape[1], 1))
        #print(image.shape)

        image = np.divide(image, 255)
        np.save("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/arrays/{}_r.npy".format(path.split('/')[-1].split('_')[0]), image)

    for path in dataset['I5']:
        image = cv2.imread("/home/martin/data/v2/filtered/scaled/{}".format(path.split('/')[-1]))
        image = image[:, :, 1]
        image = image.reshape((image.shape[0]*image.shape[1], 1))
        image = np.divide(image, 255)
        np.save("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/arrays/{}_l.npy".format(path.split('/')[-1].split('_')[0]), image)

def prepareNewDataset():
    dataset = readDataset('/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/dataset_flt.csv')
    for idx in range(dataset['I4'].shape[0]):
        dataset['I4'][idx] = "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/arrays/{}_r.npy".format(dataset['I4'][idx].split('/')[-1].split('_')[0])

    for idx in range(dataset['I5'].shape[0]):
        dataset['I5'][idx] = "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/arrays/{}_l.npy".format(dataset['I5'][idx].split('/')[-1].split('_')[0])

    X, Y = datasetToXY(dataset)
    X_trn, Y_trn, X_tst, Y_tst = splitDataSetToTestAndTrain(X, Y, ratio=0.9)
    trainX = pd.DataFrame(X_trn)
    trainY = pd.DataFrame(Y_trn)
    testX = pd.DataFrame(X_tst)
    testY = pd.DataFrame(Y_tst)

    trainX.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/x_train.csv")
    trainY.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/y_train.csv")
    testX.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/x_test.csv")
    testY.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/y_test.csv")

def inputCoder(inputs, neurons, widths):
    titlPeaks = uniformPeakPlacement(-20, 10, neurons[0])
    versionPeaks = uniformPeakPlacement(-30, 30, neurons[1])
    vergencePeaks = uniformPeakPlacement(17, 41, neurons[2])
    xPeaks = uniformPeakPlacement(0, 320, neurons[3])
    yPeaks = uniformPeakPlacement(0, 280, neurons[4])
    # sizePeaks = uniformPeakPlacement(50, 2050, 42)

    titl = gaussianPopulationCoding(inputs[0], titlPeaks, w=widths[0], h=1)
    version = gaussianPopulationCoding(inputs[1], versionPeaks, w=widths[1], h=1)
    vergence = gaussianPopulationCoding(inputs[2], vergencePeaks, w=widths[2], h=1)
    if len(inputs) > 3:
        x1 = gaussianPopulationCoding(inputs[3], xPeaks, w=widths[3], h=1)
        x2 = gaussianPopulationCoding(inputs[6], xPeaks, w=widths[3], h=1)
        y1 = gaussianPopulationCoding(inputs[4], yPeaks, w=widths[4], h=1)
        y2 = gaussianPopulationCoding(inputs[7], yPeaks, w=widths[4], h=1)
#       s1 = gaussianPopulationCoding(inputs[5], sizePeaks, w=42, h=1)
#       s2 = gaussianPopulationCoding(inputs[8], sizePeaks, w=50, h=1)

        #WARNING: consider adding of sizes
        return titl, version, vergence, x1, y1, x2, y2

    return titl, version, vergence

def outputCoder(outputs, neurons, widths):
    o1Peaks = uniformPeakPlacement(-95, -63, neurons[0])
    o2Peaks = uniformPeakPlacement(22, 26, neurons[1])
    o3Peaks = uniformPeakPlacement(18, 80, neurons[2])
    o4Peaks = uniformPeakPlacement(60, 107, neurons[3])
    o5Peaks = uniformPeakPlacement(-90, 90, neurons[4])
    o6Peaks = uniformPeakPlacement(-21, 1, neurons[5])
    o7Peaks = uniformPeakPlacement(-21, 7, neurons[6])

    o1 = gaussianPopulationCoding(outputs[0], o1Peaks, w=widths[0], h=1)
    o2 = gaussianPopulationCoding(outputs[1], o2Peaks, w=widths[1], h=1)
    o3 = gaussianPopulationCoding(outputs[2], o3Peaks, w=widths[2], h=1)
    o4 = gaussianPopulationCoding(outputs[3], o4Peaks, w=widths[3], h=1)
    o5 = gaussianPopulationCoding(outputs[4], o5Peaks, w=widths[4], h=1)
    o6 = gaussianPopulationCoding(outputs[5], o6Peaks, w=widths[5], h=1)
    o7 = gaussianPopulationCoding(outputs[6], o7Peaks, w=widths[6], h=1)

    return o1, o2, o3, o4, o5, o6, o7

def loadNewDataset():
    trainX = pd.read_csv(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/x_train.csv")
    trainY = pd.read_csv(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/y_train.csv")
    testX=pd.read_csv(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/x_test.csv")
    testY=pd.read_csv(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/y_test.csv")

    trainX = trainX.values
    trainXNew = list()
    for i in range(trainX.shape[0]):
        r_img = np.load(trainX[i][4])
        l_img = np.load(trainX[i][5])

        titl = trainX[i][1]
        version = trainX[i][2]
        vergence = trainX[i][3]

        titl, version, vergence = inputCoder([titl, version, vergence])
        eyesInf = titl + version + vergence
        retinal = np.concatenate((r_img, l_img)).reshape((r_img.shape[0]+l_img.shape[0],)).tolist()
        trainXNew.append(eyesInf + retinal)

    testX = testX.values
    testXNew = list()
    for i in range(testX.shape[0]):
        r_img = np.load(testX[i][4])
        l_img = np.load(testX[i][5])

        titl = testX[i][1]
        version = testX[i][2]
        vergence = testX[i][3]

        titl, version, vergence = inputCoder([titl, version, vergence])
        eyesInf = titl + version +  vergence
        retinal = np.concatenate((r_img, l_img)).reshape((r_img.shape[0]+l_img.shape[0],)).tolist()
        testXNew.append(eyesInf + retinal)

    trainY = trainY.values
    trainYNew=list()
    for idx in range(trainY.shape[0]):
        o1 = trainY[idx][1]
        o2 = trainY[idx][2]
        o3 = trainY[idx][3]
        o4 = trainY[idx][4]
        o5 = trainY[idx][5]
        o6 = trainY[idx][6]
        o7 = trainY[idx][7]

        o1, o2, o3, o4, o5, o6, o7 = outputCoder([o1, o2, o3, o4, o5, o6, o7])
        output = o1 + o2 + o3 + o4 + o5 + o6 + o7
        trainYNew.append(output)

    testY = testY.values
    testYNew = list()
    for idx in range(testY.shape[0]):
        o1 = testY[idx][1]
        o2 = testY[idx][2]
        o3 = testY[idx][3]
        o4 = testY[idx][4]
        o5 = testY[idx][5]
        o6 = testY[idx][6]
        o7 = testY[idx][7]

        o1, o2, o3, o4, o5, o6, o7 = outputCoder([o1, o2, o3, o4, o5, o6, o7])
        output = o1 + o2 + o3 + o4 + o5 + o6 + o7
        testYNew.append(output)

    return np.array(trainXNew), np.array(trainYNew), np.array(testXNew), np.array(testYNew)

def loadNewDataset1(inpNeurons, inpWidhts, outNeurons, outWidhts):
    dataset = readDataset("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/new_flt.csv")
    X, Y = datasetToXY(dataset)
    xtrn, ytrn, xtes, ytes = splitDataSetToTestAndTrain(X, Y, ratio=0.80)

    np.save("x_test_last.npy", xtes)
    np.save("y_test_last.npy", ytes)

    xtrnN = list()
    ytrnN = list()
    xtesN = list()
    ytestN = list()

    for i in range(xtrn.shape[0]):
        #WARNING: Consider adding of sizes
        #if i == 5:
            #print(xtrn[i])
            #print(ytrn[i])
        titl, version, vergence, x1, y1, x2, y2 = inputCoder(xtrn[i], inpNeurons, inpWidhts)
        inp = titl + version + vergence + x1 + y1 + x2 + y2
        xtrnN.append(inp)

        o1, o2, o3, o4, o5, o6, o7 = outputCoder(ytrn[i], outNeurons, outWidhts)
        inp = o1 + o2 + o3 + o4 + o5 + o6 + o7
        ytrnN.append(inp)

    for i in range(xtes.shape[0]):
        titl, version, vergence, x1, y1, x2, y2 = inputCoder(xtes[i], inpNeurons, inpWidhts)
        inp = titl + version + vergence + x1 + y1 + x2 + y2
        xtesN.append(inp)

        o1, o2, o3, o4, o5, o6, o7 = outputCoder(ytes[i], outNeurons, outWidhts)
        inp = o1 + o2 + o3 + o4 + o5 + o6 + o7
        ytestN.append(inp)

    return np.array(xtrnN), np.array(ytrnN), np.array(xtesN), np.array(ytestN)

'''
Decoding output from population coding back to real values
Outputs: Particular hand angles
'''
def decodeOutput(output, neurons, widths):
    o1Peaks = uniformPeakPlacement(-95, -63, neurons[0])
    o2Peaks = uniformPeakPlacement(22, 26, neurons[1])
    o3Peaks = uniformPeakPlacement(18, 80, neurons[2])
    o4Peaks = uniformPeakPlacement(60, 107, neurons[3])
    o5Peaks = uniformPeakPlacement(-90, 90, neurons[4])
    o6Peaks = uniformPeakPlacement(-21, 1, neurons[5])
    o7Peaks = uniformPeakPlacement(-21, 7, neurons[6])

    outputs = []
    rmIdx = 0
    for i in range(len(neurons)):
        outputs.append(output[rmIdx:rmIdx+neurons[i]])
        rmIdx += neurons[i]

    o1 = gaussianPopulationDecoding(outputs[0], o1Peaks, 1, widths[0])
    o2 = gaussianPopulationDecoding(outputs[1], o2Peaks, 1, widths[1])
    o3 = gaussianPopulationDecoding(outputs[2], o3Peaks, 1, widths[2])
    o4 = gaussianPopulationDecoding(outputs[3], o4Peaks, 1, widths[3])
    o5 = gaussianPopulationDecoding(outputs[4], o5Peaks, 1, widths[4])
    o6 = gaussianPopulationDecoding(outputs[5], o6Peaks, 1, widths[5])
    o7 = gaussianPopulationDecoding(outputs[6], o7Peaks, 1, widths[6])

    return o1, o2, o3, o4, o5, o6, o7

def decodeInput(inputs, neurons, widths):
    titlPeaks = uniformPeakPlacement(-20, 10, neurons[0])
    versionPeaks = uniformPeakPlacement(-30, 30, neurons[1])
    vergencePeaks = uniformPeakPlacement(17, 41, neurons[2])
    xPeaks = uniformPeakPlacement(0, 320, neurons[3])
    yPeaks = uniformPeakPlacement(0, 280, neurons[4])

    inputsValues = []
    rmIdx = 0
    for i in range(len(neurons)):
        inputsValues.append(inputs[rmIdx:rmIdx+neurons[i]])
        rmIdx += neurons[i]

    titl = gaussianPopulationDecoding(inputsValues[0], titlPeaks, w=widths[0], h=1)
    version = gaussianPopulationDecoding(inputsValues[1], versionPeaks, w=widths[1], h=1)
    vergence = gaussianPopulationDecoding(inputsValues[2], vergencePeaks, w=widths[2], h=1)
    x1 = gaussianPopulationDecoding(inputsValues[3], xPeaks, w=widths[3], h=1)
    y1 = gaussianPopulationDecoding(inputsValues[4], yPeaks, w=widths[4], h=1)
    x2 = gaussianPopulationDecoding(inputsValues[5], xPeaks, w=widths[5], h=1)
    y2 = gaussianPopulationDecoding(inputsValues[6], yPeaks, w=widths[6], h=1)

    return titl, version, vergence, x1, y1, x2, y2

def getPredDataset2():
    datasetName = "find2"
    model = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v2np/")
    print(model.getName())

    trainX, trainY, testX, testY = loadSavedDataset(datasetName)

    inpNeurons = [10, 12, 8, 32, 28, 32, 28]
    inpWidths = [4, 5, 4, 8, 6, 8, 6]
    outNeurons = [6, 3, 10, 11, 20, 4, 5]
    outWidth = [6, 1.5, 7, 7, 10, 6, 7]
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    data = []
    for i in range(testY.shape[0]):
        codedInput = testX[i,:]
        codedOutput = testY[i,:]

        predOutput = model.predictForward(codedInput)
        predInput = model.predictBackward(codedOutput)

        decodedPredOutput = decodeOutput(predOutput[0], outNeurons, outWidth)
        decodedPredInput  = decodeInput(predInput[0], inpNeurons, inpWidths)

        out = list(decodedPredOutput)
        inp = list(decodedPredInput)[0:3]
        line = inp + out
        data.append(line)

    dataframe = pd.DataFrame(data, columns=['I0', 'I1', 'I2', 'O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6'])
    dataframe.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v2np/preDataset.csv")

def filterDataset1To1():
    dataset = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/datasetv2.2.csv")
    new_dataset = pd.DataFrame(columns=dataset.columns)

    data = dataset.values

    for sample in data:
        count = new_dataset.loc[(new_dataset["I1"] == sample[0]) & (new_dataset["I2"] == sample[2]) & (new_dataset["I3"] == sample[5])].values.shape[0]
        if count == 0:
            new_dataset = new_dataset.append(pd.Series(sample, index=new_dataset.columns), ignore_index=True)

    new_dataset.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/datasetv2.3.csv")

if __name__ == '__main__':
    #mergePointsWithPredictions(version="ret")
    #rescaleImages("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/dataset_flt.csv")
    #prepareNewDataset()
    #loadNewDataset()

    #loadNewDataset1()
    #getPredDataset2()
    #mergePointsWithPredictions(version="ret2")
    #filterDataset1To1()

    #mergePointsWithPredictions("ret")
    #pointsDistanceError("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/PointsToComparedv2.csv")
    #getPredDataset2()

    pass