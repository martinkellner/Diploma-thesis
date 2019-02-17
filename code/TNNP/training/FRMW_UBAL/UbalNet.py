import numpy as np
import matplotlib.pyplot as plt
from training.FRMW_UBAL.DataPreparation import getData, splitDataSetToTestAndTrain
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import datetime
from sklearn.externals import joblib



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

        self.af = ""
        self.alpha = None
        self.epochs = None
        self.numHidden = None
        self.f_beta = [0, .2, .8]
        self.b_beta = [.8, .2, .0]
        self.f_gamma = []
        self.b_gamma = []
        self.gamma = [0, 0, 0, .8]
        self.count = None

        self.fig, self.axs = plt.subplots(2, 1)
        self.foldername = ""
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

    def forward_prediction(self, input, weights, bias):
        return self.activation(np.dot(input, weights) + bias)

    def forward_echo(self, prediction, weights, bias):
        return self.activation(np.dot(prediction, weights) + bias)

    def backward_prediction(self, output, weights, bias):
        return self.activation(np.dot(output, weights) + bias)

    def backward_echo(self, prediction, weights, bias):
        return self.activation(np.dot(prediction, weights) + bias)

    def fit(self, x_train, y_train, x_test, y_test):
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Wrong dimensions, numbers of X and y must be the same!")

        self.count = X.shape[0]
        self.W1 = np.random.uniform(size=(x_train.shape[1], self.numHidden))
        self.b1 = np.random.uniform(size=(1, self.numHidden))
        self.W2 = np.random.uniform(size=(self.numHidden, y_train.shape[1]))
        self.b2 = np.random.uniform(1, y_train.shape[1])

        self.M1 = np.random.uniform(size=(y_train.shape[1], self.numHidden))
        self.d1 = np.random.uniform(size=(1, self.numHidden))
        self.M2 = np.random.uniform(size=(self.numHidden, x_train.shape[1]))
        self.d2 = np.random.uniform(size=(1, x_train.shape[1]))

        self.training(x_train, y_train, x_test, y_test)

    def saveWeights(self):
        np.save(self.foldername + "W_1.npy", self.W1)
        np.save(self.foldername + "W_2.npy", self.W2)
        np.save(self.foldername + "M_1.npy", self.M1)
        np.save(self.foldername + "M_2.npy", self.M2)
        np.save(self.foldername + "B_1.npy", self.b1)
        np.save(self.foldername + "B_2.npy", self.b2)
        np.save(self.foldername + "D_1.npy", self.d1)
        np.save(self.foldername + "D_2.npy", self.d2)

    def predictForward(self, x):
        hidd_step = self.forward_prediction(x, self.W1, self.b1)
        return self.forward_prediction(hidd_step, self.W2, self.b2)

    def predictBackward(self, y):
        hidd_step = self.backward_prediction(y, self.M1, self.d1)
        return self.backward_prediction(hidd_step, self.M2, self.d2)

    def test(self, test_X, test_Y):
        testXerr = []
        testYerr = []
        for i in range(test_X.shape[0]):
            testXerr.append(self.mse(test_Y[i], self.predictForward(test_X[i])))
            testYerr.append(self.mse(test_X[i], self.predictBackward(test_Y[i])))

        return np.mean(testXerr), np.mean(testYerr)

    def training(self, x_train, y_train, x_test, y_test):

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

        for ep in range(self.epochs):
            epFError = []
            epBError = []
            neuOutputErr = []
            neuInpErr = []

            np.random.shuffle(indexs)

            for idx in indexs:
                qhfp = self.forward_prediction(x_train[idx], self.W1, self.b1)
                pofe = self.forward_echo(qhfp, self.M2, self.d2)

                qofp = self.forward_prediction(qhfp, self.W2, self.b2)
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

                neuInpErr.append(np.abs(x_train[idx] - pobp))
                neuOutputErr.append(np.abs(y_train[idx] - qofp))

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
        plt.show()

    def mse(self, target, output):
        return np.sum((target - output) ** 2)

    def mao(self, targer, output):
        return np.sum(np.abs(targer - output))

    def activation(self, x):
        if self.af == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.af == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def setHyperparamenters(self, alpha, epochs, numhidden, activationn, f_beta, b_beta, gammas):
        self.alpha = alpha
        self.epochs = epochs
        self.numHidden = numhidden
        self.af = activationn
        self.b_beta = b_beta
        self.f_beta = f_beta
        self.gamma = gammas


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
    bestname = ""
    for i in range(len(results)):
        if lists:
            print("[{}] - ".format(i + 1) + results[i])
        numbers = getNumbers(results[i])
        if sum(bestScore) > sum(numbers):
            bestScore = numbers
            best = results[i]

    waitInput = 'best' #input("Select model to load: ")
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

    else:
        print("Exit")


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


def findHyperParameters():
    X, Y = getData()
    x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, Y)

    f_beta = [0, None, None]
    b_beta = [None, None, 0]
    gammas = [0, 0, 0, None]

    for aplha in [0.12]:
        for neurons in [15]:
            for epochs in [100, 200]:
                for beta in [.5, .7, .8]:
                    for gamma in [0.5, 0.7, 0.9]:
                        name = str(aplha) + "_" + str(neurons) + "_" + str(epochs) + "_" + str(beta) + "_" + str(gamma)
                        b_beta[0] = beta
                        b_beta[1] = 1 - beta
                        f_beta[1] = 1 - beta
                        f_beta[2] = beta
                        gammas[3] = gamma
                        un1 = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/", name)
                        un1.setHyperparamenters(aplha, epochs, neurons, "sigmoid", f_beta, b_beta, gammas)
                        un1.fit(x_train, y_train, x_test, y_test)

def getPredDataset(X, y, model, pathtosave=None):
    dataR = []; dataP = []
    for i in range(X.shape[0]):
        predY = model.predictForward(X[i])
        predX = model.predictBackward(y[i])
        rowR = X[i].tolist() + y[i].tolist(); rowP =  predX[0].tolist() + predY[0].tolist()
        #print(predY.tolist())
        #print(row)
        dataR.append(rowR); dataP.append(rowP)

    #print(data)
    dataR = np.array(dataR)
    dataP = np.array(dataP)

    scaler = joblib.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc1.pkl")

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


def validateModel(model, X, Y):
    _, _, real, pred = getPredDataset(X, Y, model,
                                      '/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/prediction/1.NET')
    diff = np.abs(real - pred)
    print("Avarage mean error: (abs(targer - prediction))")
    print(np.mean(diff))

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
    bestModel = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models")
    bestModel.af = "sigmoid"

    X, Y = getData()
    #x_train, y_train, x_test, y_test = splitDataSetToTestAndTrain(X, Y)
    '''
    un1 = UbalNet("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/", "best_hyp_0.08_12_500_0.5_0.9")
    un1.setHyperparamenters(0.08, 300, 12, "sigmoid", [0, .5, .5], [.5, .5, 0], [0, 0, 0, .9])
    un1.fit(x_train, y_train, x_test, y_test)
    '''

    #print(bestModel.test(x_test, y_test))
    validateModel(bestModel, X, Y)
