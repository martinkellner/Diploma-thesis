from training.FRMW_UBAL.DataPreparation import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import datetime
from sklearn.externals import joblib
from mpl_toolkits import mplot3d
import math
from training.FRMW_UBAL.populationCoder import *

class UbalNet:

    def __init__(self, path, name, loaded=False):
        self.W1 = None
        self.W2 = None
        self.M1 = None
        self.M2 = None
        self.b1 = None
        self.b2 = None
        self.d1 = None
        self.d2 = None
        self.e = None
        self.r = None
        self.threshold = None

        self.af = "sigmoid"
        self.alpha = None
        self.epochs = None
        self.numHidden = None
        self.f_beta = [0, .2, .8]
        self.b_beta = [.8, .2, .0]
        self.f_gamma = []
        self.b_gamma = []
        self.gamma = [0, 0, 0, .8]
        self.count = None

        #self.fig, self.axs = plt.subplots(2, 1)

        self.foldername = ""
        self.last_forward_activation = None
        self.last_backward_activation = None
        self.last_forward_sum = None
        self.last_backward_sum = None

        if not loaded:
            if os.path.isdir(path):
                filenames = (path if path[-1] == '/' else path + '/') + name
                timestmps = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
                self.foldername = filenames + '_' + timestmps + '/'
                os.mkdir(self.foldername)
            if self.foldername == "":
                print("Cannot create folder from given params: {}, {}".format(path, name))
        else:
            self.foldername = (path if path[-1] == '/' else path + '/') + name

    def getName(self):
        return self.foldername

    def forward_prediction(self, input, weights, bias, divided=False, threshold=0):
        sum = np.dot(input, weights) + bias
        return self.activation(sum), sum

    def forward_echo(self, prediction, weights, bias):
        return self.activation(np.dot(prediction, weights) + bias)

    def backward_prediction(self, output, weights, bias):
        sum = np.dot(output, weights) + bias
        return self.activation(sum)

    def backward_echo(self, prediction, weights, bias):
        return self.activation(np.dot(prediction, weights) + bias)

    def fit(self, x_train, y_train, x_test, y_test, log=True):
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Wrong dimensions, numbers of X and y must be the same!")

        self.W1 = np.random.uniform(size=(x_train.shape[1], self.numHidden))
        self.b1 = np.random.uniform(size=(1, self.numHidden))
        self.W2 = np.random.uniform(size=(self.numHidden, y_train.shape[1]))
        self.b2 = np.random.uniform(1, y_train.shape[1])

        self.M1 = np.random.uniform(size=(y_train.shape[1], self.numHidden))
        self.d1 = np.random.uniform(size=(1, self.numHidden))
        self.M2 = np.random.uniform(size=(self.numHidden, x_train.shape[1]))
        self.d2 = np.random.uniform(size=(1, x_train.shape[1]))

        self.training(x_train, y_train, x_test, y_test, log=log)

    def saveWeights(self):
        np.save(self.foldername + "W_1.npy", self.W1)
        np.save(self.foldername + "W_2.npy", self.W2)
        np.save(self.foldername + "M_1.npy", self.M1)
        np.save(self.foldername + "M_2.npy", self.M2)
        np.save(self.foldername + "B_1.npy", self.b1)
        np.save(self.foldername + "B_2.npy", self.b2)
        np.save(self.foldername + "D_1.npy", self.d1)
        np.save(self.foldername + "D_2.npy", self.d2)

    def predictForward(self, x, divided=False, threshold=None, e=None, r=None):
        if divided:
            self.e = e
            self.r = r
        hidd_step, sum = self.forward_prediction(x, self.W1, self.b1, divided=divided, threshold=threshold)
        self.last_forward_activation = hidd_step
        self.last_forward_sum = sum
        return self.forward_prediction(hidd_step, self.W2, self.b2)[0]

    def predictBackward(self, y):
        hidd_step = self.backward_prediction(y, self.M1, self.d1)
        self.last_backward_activation = hidd_step
        return self.backward_prediction(hidd_step, self.M2, self.d2)

    def test(self, test_X, test_Y):
        testXerr = []
        testYerr = []
        for i in range(test_X.shape[0]):

            testXerr.append(self.mse(test_Y[i], self.predictForward(test_X[i])))
            testYerr.append(self.mse(test_X[i], self.predictBackward(test_Y[i])))

        return np.mean(testXerr), np.mean(testYerr)

    def training(self, x_train, y_train, x_test, y_test, log):
        fError = []
        bError = []
        indexs = np.arange(x_train.shape[0])
        epTFError = []
        epTBError = []
        totalAvgError = 9999

        avgErrorNeuronOutput = []
        avgErrorNeuronInput = []

        logfile = open(self.foldername + 'logfile.txt', 'w+')
        print(self.alpha, self.af, self.epochs, file=logfile)
        print("Training - ", self.alpha, self.af, self.epochs, self.f_beta, self.b_beta, self.gamma)

        for ep in range(self.epochs):
            epFError = []
            epBError = []
            neuOutputErr = []
            neuInpErr = []

            np.random.shuffle(indexs)

            for idx in indexs:

                if self.e is not None and self.r is not None:
                    qhfp = self.forward_prediction(x_train[idx], self.W1, self.b1, divided=True, threshold=self.threshold)[0]
                else:
                    qhfp = self.forward_prediction(x_train[idx], self.W1, self.b1)[0]
                pofe = self.forward_echo(qhfp, self.M2, self.d2)

                qofp = self.forward_prediction(qhfp, self.W2, self.b2)[0]
                phfe = self.forward_echo(qofp, self.M1, self.d1)

                phbp = self.backward_prediction(y_train[idx], self.M1, self.d1)
                qobe = self.backward_echo(phbp, self.W2, self.b2)

                pobp = self.backward_prediction(phbp, self.M2, self.d2)
                qhbe = self.backward_echo(pobp, self.W1, self.b1)

                tfh = self.f_beta[1] * qhfp + (1 - self.b_beta[1]) * phbp
                efh = self.gamma[3] * qhfp + (1 - self.gamma[3]) * qhbe
                tbh = self.b_beta[1] * phbp + (1 - self.f_beta[1]) * qhfp
                ebh = self.gamma[3] * phbp + (1 - self.gamma[3]) * phfe

                tfo = self.f_beta[2] * qofp + (1 - self.f_beta[2]) * y_train[idx]
                efo = self.gamma[3] * qofp + (1 - self.gamma[3]) * qobe
                tbo = self.b_beta[0] * pobp + (1 - self.b_beta[0]) * x_train[idx]
                ebo = self.gamma[3] * pobp + (1 - self.gamma[3]) * pofe

                # update weights and biases
                delta_h_f = self.alpha * np.dot(tbo.T, (tfh - efh))
                self.W1 += delta_h_f
                self.b1 += np.sum(tfh - efh, axis=0)

                delta_o_f = self.alpha * np.dot(tbh.T, (tfo - efo))
                self.W2 += delta_o_f
                self.b2 += np.sum(tfo - efo, axis=0)

                delta_h_b = self.alpha * np.dot(tfo.T, (tbh - ebh))
                self.M1 += delta_h_b
                self.d1 += np.sum(tbh - ebh, axis=0)

                delta_o_b = self.alpha * np.dot(tfh.T, (tbo - ebo))
                self.M2 += delta_o_b
                self.d2 += np.sum(tbo - ebo, axis=0)

                epFError.append(self.mse(y_train[idx], qofp))
                epBError.append(self.mse(x_train[idx], pobp))
                #neuInpErr.append(np.abs(x_train[idx] - pobp))
                #neuOutputErr.append(np.abs(y_train[idx] - qofp))

            fError.append(np.mean(epFError))
            bError.append(np.mean(epBError))

            avgErrorNeuronInput.append(np.mean(neuInpErr, axis=0))
            avgErrorNeuronOutput.append(np.mean(neuOutputErr, axis=0))

            testFerr, testBerr = self.test(x_test, y_test)

            avgError = (fError[-1] + bError[-1] + testFerr + testBerr) / 4
            save = False
            if avgError < totalAvgError:
                self.saveWeights()
                totalAvgError = avgError
                save = True

            epTFError.append(testFerr)
            epTBError.append(testBerr)
            if log:
                print(
                    "Epoch: {}\tTRAINING:\tForward MSE:{}\tBackward MSE:{}\tTESTING:\tForward MSE:{}\tBackward MSE:{}\tSAVED:{}".format(
                        ep, fError[-1], bError[-1], testFerr, testBerr, save))
            print(
            "Epoch: {}\tTRAINING:\tForward MSE:{}\tBackward MSE:{}\tTESTING:\tForward MSE:{}\tBackward MSE:{}\tSAVED:{}".format(
                ep, fError[-1], bError[-1], testFerr, testBerr, save), file=logfile)

        self.plotError(fError, bError, epTFError, epTBError, saveFig=True)
        # self.plotIOError(np.array(avgErrorNeuronInput), np.array(avgErrorNeuronOutput))

    def plotIOError(self, inputDiff, outputDiff):
        fig, axs = plt.subplots(2, 1)
        for i in range(inputDiff.shape[2]):
            axs[0].plot(np.arange(inputDiff.shape[0]), inputDiff[:, :, i], label=str(i + 1) + ". input")
        for j in range(outputDiff.shape[2]):
            axs[1].plot(np.arange(outputDiff.shape[0]), outputDiff[:, :, j], label=str(j + 1) + ". output")
        plt.show()

    def plotError(self, errF, errB, TerrF, TerrB, saveFig=False):
        fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2)

        axs1.plot(np.arange(len(errF)), errF)
        axs1.set_title("Forward MSE error")
        axs1.set_xlabel("Epochs")
        axs1.set_ylabel("FP MSE")

        axs2.plot(np.arange(len(errB)), errB)
        axs2.set_title("Backward MSE error")
        axs2.set_xlabel("Epochs")
        axs2.set_ylabel("BP MSE")

        axs3.plot(np.arange(len(TerrF)), TerrF)
        axs3.set_title("Forward MSE error - TEST")
        axs3.set_xlabel("Epochs")
        axs3.set_ylabel("FP MSE")

        axs4.plot(np.arange(len(TerrB)), TerrB)
        axs4.set_title("Backward MSE error - TEST")
        axs4.set_xlabel("Epochs")
        axs4.set_ylabel("BP MSE")

        if saveFig:
            plt.savefig(self.foldername + "figure.png")

    def mse(self, target, output):
        return np.sum((target - output) ** 2)

    def mao(self, targer, output):
        return np.sum(np.abs(targer - output))

    def activation(self, x):
        if self.af == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.af == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def setHyperparamenters(self, alpha, epochs, numhidden, activationn, f_beta, b_beta, gammas, r=None, e=None, threshold=None):
        self.alpha = alpha
        self.epochs = epochs
        self.numHidden = numhidden
        self.af = activationn
        self.b_beta = b_beta
        self.f_beta = f_beta
        self.gamma = gammas
        self.e = e
        self.r = r
        self.threshold = threshold

    def get_last_activations(self, direction="Forward"):
        return self.last_forward_activation[0]

    def get_last_sums(self, direction="Forward"):
        return self.last_forward_sum[0]

def getListOfBestError(modelsfolder):
    logfile = "logfile.txt"
    bestResults = []

    if os.path.isdir(modelsfolder):
        subdirs = [x[0] for x in os.walk(modelsfolder)]
        for sdir in subdirs:
            filename = sdir + "/" + logfile
            if os.path.isfile(filename):
                with open(filename, 'r') as file:
                    lastTrue = ""
                    for line in file:
                        if line.endswith("True\n"):
                            lastTrue = line
                result = sdir.split('/')[-1] + " " + lastTrue
                bestResults.append(result)

    return bestResults


def loadModel(foldername, lists=False):
    results = getListOfBestError(foldername)
    if len(results) == 0:
        print("No models! EXIT!")
        return
    best = ""
    bestScore = [9999]
    bestFBScore = [9999]
    bestname = ""
    bestFName = ""
    for i in range(len(results)):
        if lists:
            print("[{}] - ".format(i + 1) + results[i])
        numbers = getNumbers(results[i])

        if sum(bestScore) > sum(numbers):
            bestScore = numbers
            best = results[i]
        try:
            if (numbers[0] + numbers[2])<sum(bestFBScore):
                bestFBScore = [numbers[0], numbers[2]]
                bestFName = results[i]
        except:
            continue

    waitInput = "best" #input("Select model to load: ")
    if waitInput.isdigit():
        pass
    elif waitInput == "best":
        print(best, bestScore)
        name = best[0:best.find(' ')]
        dir = (foldername + name if foldername[-1] == '/' else foldername + '/' + name) + '/'
        ldNet = UbalNet(foldername, name, loaded=True)
        ldNet.W1 = np.load(dir + "W_1.npy")
        ldNet.W2 = np.load(dir + "W_2.npy")
        ldNet.M1 = np.load(dir + "M_1.npy")
        ldNet.M2 = np.load(dir + "M_2.npy")
        ldNet.b1 = np.load(dir + "B_1.npy")
        ldNet.b2 = np.load(dir + "B_2.npy")
        ldNet.d1 = np.load(dir + "D_1.npy")
        ldNet.d2 = np.load(dir + "D_2.npy")

        return ldNet
    elif waitInput == "best f":
        print(bestFName, bestFBScore)
        name = bestFName[0:bestFName.find(' ')]
        dir = (foldername + name if foldername[-1] == '/' else foldername + '/' + name) + '/'
        ldNet = UbalNet(foldername, name, loaded=True)
        ldNet.W1 = np.load(dir + "W_1.npy")
        ldNet.W2 = np.load(dir + "W_2.npy")
        ldNet.M1 = np.load(dir + "M_1.npy")
        ldNet.M2 = np.load(dir + "M_2.npy")
        ldNet.b1 = np.load(dir + "B_1.npy")
        ldNet.b2 = np.load(dir + "B_2.npy")
        ldNet.d1 = np.load(dir + "D_1.npy")
        ldNet.d2 = np.load(dir + "D_2.npy")

        return ldNet

    else:
        name = waitInput
        dir = (foldername + name if foldername[-1] == '/' else foldername + '/' + name) + '/'
        ldNet = UbalNet(foldername, name, loaded=True)
        ldNet.W1 = np.load(dir + "W_1.npy")
        ldNet.W2 = np.load(dir + "W_2.npy")
        ldNet.M1 = np.load(dir + "M_1.npy")
        ldNet.M2 = np.load(dir + "M_2.npy")
        ldNet.b1 = np.load(dir + "B_1.npy")
        ldNet.b2 = np.load(dir + "B_2.npy")
        ldNet.d1 = np.load(dir + "D_1.npy")
        ldNet.d2 = np.load(dir + "D_2.npy")

        return ldNet


def getNumbers(line):
    numbers = []
    keywords = ["Forward", "Backward"]
    j = 0
    while True:
        idx = line.find(keywords[j % 2])
        if idx == -1:
            break
        number = ""
        for i in range(idx, len(line)):
            if line[i] == '\t':
                line = line[i:]
                break
            if line[i].isdigit() or line[i] == '.':
                number += line[i]
        numbers.append(number)
        j += 1

    return [float(x) for x in numbers]

def findHyperParameters(x_train, y_train, x_test, y_test):

    dfTestX = pd.DataFrame(x_test)
    dfTestY = pd.DataFrame(y_test)

    dfTestX.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/New_testX_scl3.csv")
    dfTestY.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/New_testY_scl3.csv")

    f_beta = [0, None, None]
    b_beta = [None, None, 0]
    gammas = [0, 0, 0, None]

    for aplha in [0.08]:
        for neurons in [15, 17, 20, 25]:
            for beta in [.2, .5, .7, 0.9]:
                for gamma in [0.25, 0.5, 0.75, 0.9]:
                    name = str(aplha) + "_" + str(neurons) + "_" + str(200) + "_" + str(beta) + "_" + str(gamma)
                    #print("Training - " + name)
                    b_beta[0] = beta
                    b_beta[1] = 1 - beta
                    f_beta[1] = 1 - beta
                    f_beta[2] = beta
                    gammas[3] = gamma
                    un1 = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/1-1v2/", name)
                    un1.setHyperparamenters(aplha, 200, neurons, "sigmoid", f_beta, b_beta, gammas)
                    un1.fit(x_train, y_train, x_test, y_test, log=True)

                    hyperparams = [aplha, neurons, beta, gamma]

                    val = validateModel(un1, x_test, y_test, version="1-1")
                    saveValidation(un1.getName(), validation=val, hyperparams=hyperparams, populationParams=[],
                                   datasetName="1-1v4",
                                   filename="/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/val1-1.txt")

def getPredDataset(X, y, model, pathtosave=None, version="ret"):
    if version == "1-1":
        dataR = []
        dataP = []
        for i in range(X.shape[0]):
            predY = model.predictForward(X[i])
            predX = model.predictBackward(y[i])
            rowR = X[i].tolist() + y[i].tolist()
            rowP =  predX[0].tolist() + predY[0].tolist()
            dataR.append(rowR)
            dataP.append(rowP)

        dataR = np.array(dataR)
        dataP = np.array(dataP)

        scaler = joblib.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc5.pkl")

        unSclDataR = scaler.inverse_transform(dataR)
        unSclDataP = scaler.inverse_transform(dataP)

        dfR = pd.DataFrame(dataR, columns=["I" + str(i) for i in range(X.shape[1])] + ["O" + str(i) for i in range(y.shape[1])])
        dfP = pd.DataFrame(dataP, columns=["I" + str(i) for i in range(X.shape[1])] + ["O" + str(i) for i in range(y.shape[1])])
        udfP = pd.DataFrame(unSclDataP, columns=["I" + str(i) for i in range(X.shape[1])] + ["O" + str(i) for i in range(y.shape[1])])
        udfR = pd.DataFrame(unSclDataR, columns=["I" + str(i) for i in range(X.shape[1])] + ["O" + str(i) for i in range(y.shape[1])])

        if pathtosave is not None:
            dfP.to_csv(pathtosave + "_predicted.csv")
            dfR.to_csv(pathtosave + "_real.csv")
            udfP.to_csv(pathtosave + "_predicted_u.csv")
            udfR.to_csv(pathtosave + "_real.csv_u")

        return dfR, dfP, udfR, udfP

    elif version == "ret":
        if X.shape[0] != y.shape[0]:
            print("Number of X a Y samples must be equal! And it was X:{}, Y:{}".format(X.shape[0], y.shape[0]))
            return

        yPred = []
        xPred = []
        for i in range(X.shape[0]):
            yPred.append(model.predictForward(X[i, :]))
            xPred.append(model.predictBackward(y[i, :]))
        yPred = np.array(yPred)
        xPred = np.array(xPred)
        yPred = yPred[:, 0, :]
        xPred = xPred[:, 0, :]

        print(xPred.shape, yPred.shape)

        predDataset = np.concatenate((xPred, yPred), axis=1)
        dataset = np.concatenate((X, y), axis=1)
        scaler = joblib.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc3.pkl")
        invSclPredDataset = scaler.inverse_transform(predDataset)
        invSclDataset = scaler.inverse_transform(dataset)
        datasetDf = pd.DataFrame(invSclDataset, columns=['I0','I1','I2','I3','I4','I5','I6','I7','I8','O0','O1','O2','O3','O4','O5','O6'])
        predDatasetDf = pd.DataFrame(invSclPredDataset, columns=['I0','I1','I2','I3','I4','I5','I6','I7','I8','O0','O1','O2','O3','O4','O5','O6'])

        datasetDf.to_csv(pathtosave + "real_datasetv2.csv")
        predDatasetDf.to_csv(pathtosave + "pred_datasetv2.csv")
        return  invSclDataset, invSclPredDataset

def validateModel(model, X, Y, version):
    if version == "1-1":
        _, _, real, pred = getPredDataset(X, Y, model, '/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/1-1v2/', version=version)
        diff = np.abs(real - pred)
        print("Avarage mean error: (abs(targer - prediction))")
        diff = np.mean(diff)
        print(diff)
        return diff, np.sum(diff[0:3]), np.sum(diff[3:]), np.sum(diff)

    elif version == "ret":
        dataset, predDataset = getPredDataset(X, Y, pathtosave="/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/", model=model, version="ret")
        print("### Avarage error for each features:")
        for i in range(dataset.shape[1]):
            if i < 9:
                print("I{} - abs err: {} ".format(i+1, np.mean(np.abs(dataset[:, i] - predDataset[:, i]))))
            else:
                print("O{} - abs err: {} ".format(i-8, np.mean(np.abs(dataset[:, i] - predDataset[:, i]))))

        print("### MSE for forward and backward")

def pointsDistanceError(path):
    points = pd.read_csv(path)
    fixPoints = points[['f1', 'f2', 'f3']].values
    palmPoints = points[['f1', 'f2', 'f3']].values + points[['x1', 'x2', 'x3']].values

    predHead = points[['h1', 'h2', 'h3']].values
    predArm = points[['a1', 'a2', 'a3']].values

    diffHead = []
    diffArm = []
    diffPalm = []

    print(fixPoints.shape, predArm.shape, predHead.shape, palmPoints.shape)

    for i in range(fixPoints.shape[0]):
        diffHead.append(np.linalg.norm(fixPoints[i] - predHead[i]))
        diffArm.append(np.linalg.norm(fixPoints[i] - predArm[i]))
        diffPalm.append(np.linalg.norm(palmPoints[i] - predArm[i]))

    avgErrArm = np.around(np.mean(diffArm), 4)
    avgErrHead = np.around(np.mean(diffHead), 4)
    avgErrHeadPalm = np.around(np.mean(diffPalm), 4)

    print("Arg distance error against fix point - backward: {} m.".format(avgErrHead))
    print("Arg distance error against fix point - forward: {} m.".format(avgErrArm))
    print("Arg distance error against palm points - forward: {} m".format(avgErrHeadPalm))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    colors = ["#890000", "#e50000", "#ef6666", "#f7b2b2"]
    count = [0, 0, 0, 0]
    color = None
    for i in range(fixPoints.shape[0]):
        if diffArm[i] <= avgErrArm/2:
            color = 3
            count[0] += 1
        elif avgErrArm/2 < diffArm[i] and diffArm[i] <= avgErrArm:
            color = 2
            count[1] += 1
        elif avgErrArm < diffArm[i] and diffArm[i] <= avgErrArm*1.5:
            color = 1
            count[2] += 1
        else:
            color = 0
            count[3] += 1
        ax.scatter(fixPoints[i, 0], fixPoints[i, 1], fixPoints[i, 2], color=colors[color])

    print("The number of test under: {} - {}\nThe number of test under: {} - {}\nThe number of test under: {} - {}\nThe number of test over {} - {}\n".format(avgErrArm/2, count[0], avgErrArm, count[1], avgErrArm*1.5, count[2], avgErrArm*1.5, count[3]))
    print("MAX error: {}, MIN Error:{}".format(max(diffArm), min(diffArm)))
    # Uncomment to show the error 3D points plot
    plt.show()
    plt.figure()
    diffHead = np.multiply(diffHead, 100)
    maxFixAngHeadError = math.ceil(max(diffHead))

    n, bins, patches = plt.hist(diffHead, bins=np.arange(0, maxFixAngHeadError, 1), facecolor='blue', alpha=0.5)
    plt.xlabel('Distance error (cm)')
    plt.ylabel('Number of points')
    plt.title(r'Histogram of Distance Error: Gaze fixation points')
    plt.savefig("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/fixHist.png")

    diffPalm = np.multiply(diffPalm, 100)
    maxFixAngPalmError = math.ceil(max(diffPalm))
    fig = plt.figure()
    n, bins, patches = plt.hist(diffPalm, bins=np.arange(0, maxFixAngPalmError, 1), facecolor='blue', alpha=0.5)
    plt.xlabel('Distance error (cm)')
    plt.ylabel('Number of points')
    plt.title(r'Histogram of Distance Error: Points of the centre of palm')
    plt.savefig(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/fixPalm.png")

def validateNewModel(testX, testY, model, inpNeurons, intWidth, outNeurons, outWidth):
    preds = list()
    reals = list()

    for i in range(testX.shape[0]):
        real = decodeOutput(testY[i], neurons=outNeurons, widths=outWidth)
        pred = decodeOutput(model.predictForward(testX[i])[0], neurons=outNeurons, widths=outWidth)

        preds.append(pred)
        reals.append(real)

    agrserr = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(preds)):
        for j in range(7):
            agrserr[j] += np.abs([reals[i][j]- preds[i][j]])

    for i in range(7):
        agrserr[i] /= len(preds)

    return agrserr

def saveDataset(xtrn, ytrn, xtes, ytes, name):
    path = "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/saved_dataset/" + name
    np.save(path + "_{}.npy".format("xtrn"), xtrn)
    np.save(path + "_{}.npy".format("ytrn"), ytrn)
    np.save(path + "_{}.npy".format("xtes"), xtes)
    np.save(path + "_{}.npy".format("ytes"), ytes)

def loadSavedDataset(name):
    path = "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/saved_dataset/" + name
    xtrn = np.load(path + "_{}.npy".format("xtrn"))
    ytrn = np.load(path + "_{}.npy".format("ytrn"))
    xtes = np.load(path + "_{}.npy".format("xtes"))
    ytes = np.load(path + "_{}.npy".format("ytes"))

    return xtrn, ytrn, xtes, ytes

def saveValidation(modelName, hyperparams, populationParams, validation, datasetName, filename):

    saveline = ""
    saveline += modelName
    for parm in hyperparams:
        saveline += " " + parm.__repr__()

    for parm in populationParams:
        saveline += " " + parm.__repr__()

    saveline += " " + datasetName
    saveline += " " + validation.__repr__()

    with open(filename, "a") as file:
        print(saveline, file=file)


if __name__ == '__main__':
    '''
    x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, y)
    model = Sequential()
    model.add(Dense(12, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    history = model.fit(x_train, y_train, epochs=150, batch_size=50, verbose=1, validation_split=0.2)
    '''
    #bestModel = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models")
    #bestModel.af = "sigmoid"

    #X, Y = getData("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/datasetv2.3.csv")
    #x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, Y)
    #findHyperParameters(x_train, y_train, x_test, y_test)

    #un1 = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/", "best_hyp_0.08_12_500_0.5_0.9")
    #un1.setHyperparamenters(0.08, 300, 12, "sigmoid", [0, .5, .5], [.5, .5, 0], [0, 0, 0, .9])
    #un1.fit(x_train, y_train, x_test, y_test)

    #x_test = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/New_testX_scl2.csv").values[:,1:]
    #y_test = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/New_testY_scl2.csv").values[:,1:]

    #bestModel = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/1-1v2/")
    #bestModel.af = "sigmoid"

    #validateModel(bestModel, x_test, y_test, version="1-1")
    #X, Y = getData("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/datasetv2.3.csv")
    #x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, Y, ratio=0.85)
    #print(x_test, y_test)
    #saveDataset(x_train, y_train, x_test, y_test, "1-1v4")
    #x_train, y_train, x_test, y_test = loadSavedDataset("1-1v4")
    #findHyperParameters(x_train, y_train, x_test, y_test)

    #/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/1-1v2/0.1_20_200_0.2_0.9_2019-04-10_19:13:36/
    #model = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/1-1v2/")
    #print(model.getName())

    #validateModel(model, x_test, y_test, version="1-1")

    '''Validation'''
    #model = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret2/")
    #print(model.getName())
    #model.test(X, y)
    ### Model name: best_hyp_0.08_12_500_0.5_0.9_2019-02-18_15:40:07
    #validateModel(model, None, None, version="ret")
    #pointsDistanceError()

    # find hyperparameters for ret version
    #findHyperParameters()

    #Receive inverse-scaled dataset of real and prediction values
    ''' Get predicted values - ret model
    model = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret/", lists=True)
    dfRealX = pd.read_csv(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/3.NET_RET-testX.csv")
    dfRealY = pd.read_csv(
        "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/3.NET_RET-testY.csv")

    X = dfRealX.values[:, 1:]
    Y = dfRealY.values[:, 1:]
    getPredDataset(X, Y, model, pathtosave="/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/", version="ret")
    '''

    ### RET MODEL using new dataset
    #X, Y = getData("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/new_flt.csv", scale=True)
    #x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, Y, ratio=0.75)
    #findHyperParameters(x_train, y_train, x_test, y_test)
    #model = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/1-1v2/")
    #X = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/New_testX_scl2.csv").values[:, 1:]
    #Y = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/New_testY_scl2.csv").values[:, 1:]
    #validateModel(model, X, Y, version='ret')
    #getPredDataset(X, Y, model, version="ret", pathtosave="/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/")
    #pointsDistanceError()

    #X, Y = getData("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/new_flt.csv", scale=False)

    #0.05_700_28_0.5_0.7
    #f_beta = [0, None, None]
    #b_beta = [None, None, 0]
    #gammas = [0, 0, 0, None]

    #beta = 0.5
    #b_beta[0] = beta
    #b_beta[1] = 1 - beta
    #f_beta[1] = 1 - beta
    #f_beta[2] = beta
    #gammas[3] = 0.7

    #alpha = 0.05
    #epochs = 700
    #neurons = 28
    #inpNeurons = [10, 12, 8, 32, 28, 32, 28]
    #inpWidths =  [5,  6,  5, 15, 13, 15, 13]

    #o1 (-95, -63)
    #o2 (22, 26)
    #o3 (18, 80)
    #o4 (60, 107)
    #o5 (-90, 90)
    #o6 (-21, 1)
    #o7 (-21, 7)
    #outNeurons = [5, 3,   7,    8,  10, 6, 7]
    #outWidth =   [8, 1.5, 10,  11,  19, 4, 5]
    #trainX, trainY, testX, testY = loadNewDataset1(inpNeurons, inpWidths, outNeurons, outWidth)
    #saveDataset(trainX, trainY, testX, testY, datasetName)
    #trainX, trainY, testX, testY = loadSavedDataset("{}".format(datasetName))

    #trainX = np.load("tranX.npy")
    #trainY = np.load("trinY.npy")
    #testX  = np.load("tesX.npy")
    #testY  = np.load("tesY.npy")
    #test = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret4/")
    #print(trainX.shape, trainY.shape)
    #test = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret4/", "test_{}_{}_{}_{}_{}".format(alpha, epochs, neurons, beta, gammas[3]))
    #test.setHyperparamenters(alpha, epochs, neurons, "sigmoid", f_beta, b_beta, gammas)
    #test.fit(trainX, trainY, testX, testY, log=True)
    # [array([6.81707768]), array([0.64855875]), array([14.52253842]), array([10.55740365]), array([17.80259667]), array([6.99817147]), array([6.38594363])]
    #validateNewModel(testX, testY, test)
    #test = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret4/", )

    #hyperparams = [alpha, epochs, neurons, f_beta, b_beta, gammas]
    #populationsParms = [inpNeurons, inpWidths, outNeurons, outWidth]
    #val = validateNewModel(testX, testY, test, inpNeurons, inpWidths, outNeurons, outWidth)
    #aveValidation(test.getName(), hyperparams, populationsParms, val, datasetName)

    '''Best reconstruction
    datasetName = "find2"

    inpNeurons = [10, 12, 8, 32, 28, 32, 28]
    inpWidths = [4, 5, 4, 8, 6, 8, 6]
    outNeurons = [6, 3, 10, 11, 20, 4, 5]
    outWidth = [6, 1.5, 7, 7, 10, 6, 7]
    trainX, trainY, testX, testY = loadNewDataset1(inpNeurons, inpWidths, outNeurons, outWidth)
    saveDataset(trainX, trainY, testX, testY, datasetName)

    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    best = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret4/", "test_{}_{}_{}_{}_{}".format(alpha, epochs, neurons, beta, gammas[3]))
    best.setHyperparamenters(alpha, epochs, neurons, "sigmoid", f_beta, b_beta, gammas)
    best.fit(trainX, trainY, testX, testY, log=True)
    val = validateNewModel(testX, testY, best, inpNeurons, inpWidths, outNeurons, outWidth)
    hyperparams = [alpha, epochs, neurons, f_beta, b_beta, gammas]
    populationsParms = [inpNeurons, inpWidths, outNeurons, outWidth]
    saveValidation(best.getName(), hyperparams, populationsParms, val, datasetName)
    '''

    inpNeurons = [10, 12, 8, 32, 28, 32, 28]
    inpWidths = [4, 5, 4, 8, 6, 8, 6]
    outNeurons = [6, 3, 10, 11, 20, 4, 5]
    outWidth = [6, 1.5, 7, 7, 10, 6, 7]

    f_beta = [0, None, None]
    b_beta = [None, None, 0]
    gammas = [0, 0, 0, None]

    aplha = 0.12
    neurons = 68
    beta = 0.7
    gamma = 0.9
    epochs = 2000
    b_beta[0] = beta
    b_beta[1] = 1 - beta
    f_beta[1] = 1 - beta
    f_beta[2] = beta
    gammas[3] = gamma

    #X, Y = getData("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/new_flt.csv")
    #x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, Y, ratio=0.85)
    #saveDataset(x_train, y_train, x_test, y_test, "find2")
    x_train, y_train, x_test, y_test = loadSavedDataset("find2")

    un1 = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/exp/", "best")
    un1.setHyperparamenters(aplha, epochs, neurons, "sigmoid", f_beta, b_beta, gammas)
    un1.fit(x_train, y_train, x_test, y_test, log=True)

    hyperparams = [aplha, epochs, neurons, f_beta, b_beta, gammas]
    populationsParms = [inpNeurons, inpWidths, outNeurons, outWidth]
    val = validateNewModel(x_test, y_test, un1, inpNeurons, inpWidths, outNeurons, outWidth)
    saveValidation(un1.getName(), hyperparams, populationsParms, val, "find2", "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/exp/exp.txt")


    #hyperparams = [aplha, neurons, beta, gamma]
    '''
    #TODO: gain fields, pozbieraj body na validaciu, zisti distance error pre vyssie model hned

    #bestModel = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1")
    #_, _, x_test, y_test = loadSavedDataset("best_retv2")
    #validateModel(bestModel, x_test, y_test, version="ret")
    #saveValidation(un1.getName(), validation=val, hyperparams=hyperparams, populationParams=[],
    #               datasetName="1-1v4",
    #               filename="/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/val1-1.txt")

    #TODO: pozbierat body pre best pc model, gain fields, hists, ...
    '''